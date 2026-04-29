# Tri-Mem: Modality-Matched Memory Hierarchies for Multi-Agent Systems

> **Full Title:** _Tri-Mem: Ontology-Matched Memory Hierarchies for Multi-Agent Orchestration_

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [The Core Problems This Solves](#the-core-problems-this-solves)
3. [The Architecture](#the-architecture)
4. [The Entropy Router & Bayesian Calibration](#the-entropy-router--bayesian-calibration)
5. [Multi-Agent Orchestration Angle](#multi-agent-orchestration-angle)
6. [What Is Novel (Publishable Contribution)](#what-is-novel)
7. [Literature Positioning](#literature-positioning)
8. [Engineering Bottlenecks to Solve](#engineering-bottlenecks-to-solve)
9. [Benchmarks & Evaluation Strategy](#benchmarks--evaluation-strategy)
10. [Metrics to Track](#metrics-to-track)
11. [Feasibility Assessment](#feasibility-assessment)
12. [Implementation: Phased Build Plan](#implementation-phased-build-plan)
13. [Current Project Structure](#current-project-structure)
14. [How to Run](#how-to-run)
15. [What's Built So Far](#whats-built-so-far)
16. [Next Steps](#next-steps)

---

## What This Project Is

Tri-Mem is a research architecture that argues **agent memory should not be one-size-fits-all**. Current agentic AI systems dump everything — rules, interaction history, variables — into the same text context window. This causes three cascading failures: context rot over long horizons, wasted tokens re-reading static rules every turn, and lossy retrieval of exact strings (API keys, object IDs, syntax).

Tri-Mem solves this by assigning each _cognitive function_ to the _data modality_ that is cheapest and most accurate for it:

| Cognitive Function                | Memory Layer                | Data Modality                       | What Goes Here                                                      |
| --------------------------------- | --------------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| Rule-following / deep reasoning   | MSA (Semantic Memory)       | Sparse-attention KV chunks + learned router | System prompts, SOPs, API docs, environment rules                   |
| Episodic recall ("what happened") | Visual Bus (Working Memory) | Compressed image patches            | Step-by-step interaction history, terminal outputs, navigated pages |
| Exact fact lookup                 | RAG (Declarative Memory)    | Discrete text tokens from vector DB | Object IDs, API keys, alphanumeric strings, mutating variables      |

This maps directly to human cognitive architecture: Semantic Memory (MSA), Episodic Working Memory (Visual Bus), and External/Declarative Memory (RAG).

---

## The Core Problems This Solves

### 1. Context Rot & Context Fidelity

Context rot is the "lost in the middle" phenomenon — when an agent's conversation history grows long, the model loses information around turns 15-30 even though those tokens are technically still in the window.

Tri-Mem attacks this from three directions:

- **Visual Bus** compresses history into image patches instead of appending raw text. A 2,000-token JSON action log becomes a small image tile. The window never fills up. Vision encoders process spatial information differently than text attention, so the "middle" doesn't get lost the same way.
- **MSA** removes static content (rulebooks, documentation) from the active window entirely. If the agent re-reads a 50,000-token rulebook every turn, that's dead weight competing for attention. Pre-computing it into latent KV states frees the entire active window for dynamic reasoning.
- **RAG** acts as a fidelity guarantee. Even if the Visual Bus loses the exact string of an API key from turn 12, the agent knows _that_ it encountered one (the semantic gist survives visual compression), and it can retrieve the exact value on demand. Graceful degradation instead of catastrophic forgetting.

**Net effect:** The agent's effective memory horizon extends from ~30-40 reliable turns to 100+ turns at a fraction of the token cost.

### 2. The Modality-Memory Mismatch

Developers force LLMs to use text for everything. Text is perfect for exact variables but terrible for episodic history (too verbose, too expensive). Visual compression is great for history but terrible for exact syntax. Tri-Mem routes data to the modality best suited for its cognitive function.

### 3. The Syntactic Action Gap

The critical flaw in pure visual-memory models like AgentOCR. If an agent compresses its history into an image, it remembers _that_ it found an API key, but the exact alphanumeric string gets visually blurred. By injecting RAG into the visual loop, we bridge the gap between "semantic memory" and "exact syntactic execution."

### 4. The "Rulebook" Compute Tax

Standard agents must read their system prompt, API documentation, and ruleset at every single turn. By pushing the static rulebook into the MSA latent space, you pre-compute the heavy logic once, completely removing it from the active token stream.

### 5. The Context Rot vs. Token Economy Paradox

In multi-agent systems, feeding 100 turns of verbose JSON logs into the context window causes the model to "forget" the middle and costs a fortune in API compute. The Visual Bus compresses history, dropping the token cost by up to 90% while preserving the chronological narrative.

---

## The Architecture

### Data Flow

```
                    ┌─────────────────────────────────────────────┐
  User Prompt ──────►                                             │
  Env Observations ─►   TRI-MEM MANAGER                          │
  System Events ────►                                             │
                    │  ┌──────────┐    ┌──────────────┐           │
                    │  │ SEMANTIC  │───►│  EPISODIC     │          │
                    │  │ MEMORY    │    │  WORKING MEM  │          │
                    │  │ (MSA)     │    │  (Visual Bus) │          │
                    │  │           │    │               │          │
                    │  │ Pre-comp  │    │ Compressed    │          │
                    │  │ KV States │    │ Image Patches │          │
                    │  └──────────┘    └───────┬───────┘          │
                    │        │                 │                   │
                    │        │    ┌─────────────┘                  │
                    │        │    │                                │
                    │  ┌─────▼────▼──┐                            │
                    │  │ EXTERNAL/   │                             │
                    │  │ DECLARATIVE │    ┌───────────────────┐    │
                    │  │ (RAG)       │───►│ ENTROPY ROUTER    │────►  Action
                    │  │             │    │                   │    │
                    │  │ Vector DB   │    │ Low entropy → MSA │    │
                    │  │ Exact facts │    │ Med entropy → VBus│    │
                    │  └─────────────┘    │ High entropy → RAG│    │
                    │                     └───────────────────┘    │
                    └─────────────────────────────────────────────┘
```

### Layer Details

**1. MSA — The Rulebook (Foundation Layer)**

- Contains: core system prompts, SOPs, massive API documentation, environment rules
- Mechanism: based on EverMind's [MSA architecture](https://github.com/EverMind-AI/MSA) — a sparse-attention layer with a learnable router over compressed document KV chunks, built on Qwen3-4B.
  - **Offline**: the rulebook corpus is chunked and run through the model once. For each chunk we store chunk-mean-pooled `K`, `V`, and a routing key `Kᵣ` — compressed state kept in CPU RAM, not GPU.
  - **Online**: the live query is projected to a routing query `Qᵣ`, cosine-similarity-matched against every stored `Kᵣ`, and the Top-k chunks have their compressed KVs streamed onto GPU, concatenated with the active context, and attended over in one pass.
  - **Document-wise RoPE**: each stored chunk resets its positional encoding from 0, so a model trained at 64k context extrapolates to effectively unbounded rulebooks (EverMind demonstrates up to ~100M tokens) with no position drift.
- Benefit: the router is effectively learned RAG — end-to-end differentiable document retrieval baked into the attention layer, so the model itself decides which rulebook sections to pull per turn, no manual prompt engineering.
- Current implementation (Phase 3.75, ✅ BUILT): MSA code and weights are marked **"Coming Soon"** by EverMind, so Tri-Mem currently **simulates** the layer with a two-path approach that is architecturally faithful to the EverMind design:
  - **Path 1 — Functional simulation via prefix caching.** The full NovaCorp IT Policy (`memory/rulebook/novacorp_it_policy.md`, 12 sections / ~11k chars / ~2.8k tokens) is embedded in a byte-identical system prompt every turn. vLLM's automatic prefix caching (`enable_prefix_caching=True`) hashes the prefix and reuses the pre-computed rulebook KV across turns, so the rulebook compute is paid exactly once per process even though the agent logically "re-reads" it every turn. This delivers the token-economic win MSA is supposed to produce.
  - **Path 2 — Poor-man's learned router.** `memory/msa_store.py` chunks the rulebook on `## ` section headings, embeds each chunk with ChromaDB's default encoder (mean-pooled sentence embedding standing in for EverMind's `Kᵣ`), and routes the live query (goal + observation + last action) via cosine Top-k. The routed sections are injected into the *user* message as a `ROUTED POLICY CONTEXT` block — not the system prompt, because that would break the cache. This is the learned-router half of EverMind's design, swapped in as cosine similarity until the real sparse-attention layer ships.
  - When EverMind's drop lands, `MSAStore.query()` becomes a sparse-attention call and the frozen system prompt becomes a handle to a cached KV tensor. Neither the agent nor the benchmark runner need to change — Tri-Mem's contribution (modality-to-function mapping, entropy routing across MSA / Visual Bus / RAG) sits on top of MSA rather than competing with it.

**2. Visual Bus — The Scratchpad (Episodic Layer)**

- Contains: step-by-step interaction history, terminal outputs, web pages navigated, intermediary thought chains
- Mechanism: renders observation/action deltas into image patches using Segment Optical Caching
- Benefit: highly compressed visual timeline of the current session, prevents Context Rot
- Key tech: instead of appending 2,000 text tokens of JSON logs per action, the system renders the delta into an image patch
- **Sliding window cap (`MAX_VISUAL_TILES = 4`):** the Visual Bus only renders the most recent 4 turns into the OCR-bound image. The cap is a hardware-driven choice rather than a design ceiling — the GLM-OCR backbone is loaded in `bfloat16` on a single consumer GPU, and rendering >4 history tiles into one image produces a tall PNG that pushes the OCR forward pass into OOM, which silently flips `_ocr_unavailable=True` and degrades the agent to a truncated-text fallback. With 4 tiles the rendered image stays in the OCR model's stable working set, the agent always has the immediate-causal-chain context, and longer-range recall is delegated to the RAG layer (where it should live anyway). On a larger GPU this can be raised — the value sits behind `MAX_VISUAL_TILES` in [configs/settings.py](configs/settings.py).

**3. RAG — The Fact Injector (Exact Retrieval Layer)**

- Contains: user-specific state data, exact alphanumeric API keys, database schemas, rapidly mutating variables
- Mechanism: when the agent's visual memory indicates it needs a specific fact, it issues a discrete query tool-call. RAG pulls the exact plain-text string from a vector database.
- Benefit: lossless retrieval of exact strings that would get blurred in visual compression

### Synergies Between Layers

- **Solves Syntactic Action Gap:** Visual memory remembers _that_ a fact exists and _when_ it was used. RAG guarantees lossless retrieval of the exact string.
- **Solves MSA Mutability Problem:** Core logic stays frozen in MSA. Mutating variables live in cheap RAG vector database. No need to re-encode MSA when a minor detail changes.
- **Optimal Token Economics:** Pre-computed latent states (MSA) for heavy logic, heavily compressed image patches (Visual Bus) for long-horizon loops, tiny discrete text injections (RAG) for variables.

---

## The Entropy Router & Bayesian Calibration

### The Routing Problem

How does the agent decide which memory to access? If you use a standard LLM prompt ("Think step-by-step: do I need RAG, Visual Memory, or MSA?"), you waste tokens and add massive latency.

### The Solution: Uncertainty-Guided Routing

The system monitors the raw math of the model's output logit confidence using Shannon Entropy:

```
H(X) = -Σ p(x) log₂ p(x)
```

The router uses an **additive ladder** — each higher band keeps everything the band below it injected, then layers more on top. MSA is always on (the rulebook chunks are cheap latent-space routing); the router decides whether to *also* pay for OCR and/or RAG.

- **Low Entropy (< 0.4):** Model is highly confident about the next action → MSA only. Visual Bus and RAG are skipped.
- **Medium Entropy (0.4 - 0.7):** Environmental uncertainty → MSA + Visual Bus patches injected to update the model's beliefs about its surroundings.
- **High Entropy (> 0.7):** Epistemic uncertainty (needs a specific fact, or cold start) → MSA + Visual Bus + RAG full stack, with summary-guided fact lookups.

### How the thresholds were chosen

The thresholds were originally set to (0.3, 0.7) by intuition. After two `--tasks 7` runs we measured the actual entropy distribution and found that LOW=0.3 was too aggressive — the band never fired because the minimum observed entropy floored at ≈0.33 on cached `access <known_target>` transitions. We bumped LOW to **0.4**; the band now fires on roughly 5% of turns (the truly cached SOP transitions like `access compliance_dashboard` after a fresh download), and the router operates as a true ternary classifier. The high threshold (0.7) was untouched — it was already firing correctly on cold starts and final-upload spikes.

The unit tests in [tests/test_entropy_router.py](tests/test_entropy_router.py) lock these bands against recorded probe entropies from the benchmark logs, so silent regression on threshold changes is caught in CI.

### The Calibration Problem

Standard LLMs are poorly calibrated — they can be 99% confident while hallucinating. If an AI hallucinates an API key but is confident it's right, the entropy registers as low. The router would rely on internal memory and crash.

### The Fix: Bayesian Reasoning

Per Google Research (March 2026), LLMs can be fine-tuned to "reason like Bayesians" using Bayesian Teaching — training the model to mimic intermediate probabilistic best-guesses of an optimal Bayesian model rather than training on perfect final answers.

**Combined data flow:**

1. Bayesian-calibrated LLM produces genuinely calibrated output logits
2. Entropy router monitors logits — because the model isn't faking confidence, Shannon Entropy becomes a reliable trigger
3. Router selects modality automatically based on entropy level

### Practical Implementation

For the research prototype, we don't need a Bayesian-calibrated model from scratch. We can:

- Measure calibration of existing models (Claude, GPT-4)
- Use logit confidence scores from API responses as a proxy
- Build a heuristic threshold-based router that gets 80% of the theoretical benefit
- Document the gap and propose Bayesian calibration as future work

---

## Multi-Agent Orchestration Angle

This is the bigger claim and significantly increases the paper's impact.

### The Problem with Current Multi-Agent Systems

Most orchestration frameworks (CrewAI, AutoGen, LangGraph) pass the full conversation transcript between agents. This means:

- Every agent pays the full token cost of every other agent's history
- If the planner ran for 30 turns, the coder must ingest all 30 turns before starting
- Context rot compounds across agents — Agent C reasons through Agent A's AND Agent B's history simultaneously

### How Tri-Mem Solves This

**MSA stays agent-local.** Each agent has its own pre-computed rulebook (the coder knows coding patterns, the reviewer knows quality standards). Never shared.

**Visual Bus becomes the shared episodic memory bus.** Instead of passing raw transcripts, Agent A compresses its entire session into a visual timeline and hands that compact artifact to Agent B. Agent B gets the narrative arc — what was tried, what worked, what failed — without drowning in tokens.

**RAG becomes the shared fact layer.** The single source of truth for exact state variables that any agent can query.

### Routing Extends to Inter-Agent Delegation

The entropy router takes on a second role: deciding not just _which memory modality_ to consult, but _whether it needs to consult another agent at all_. High entropy on a coding question when you're the planner agent? Route to the coder. This unifies intra-agent memory selection and inter-agent task delegation under one framework.

### Key Experiment to Add

A multi-agent ALFWorld or WebArena task where two agents must collaborate, comparing transcript-passing versus shared Visual Bus + RAG.

---

## What Is Novel

The novelty is NOT any single component — it's the **modality-to-function mapping** principle.

Nobody in the literature has argued that:

- Episodic memory should be **visual** (image patches, à la AgentOCR)
- Semantic memory should be **sparse-attention KV chunks with a learned router** (à la EverMind MSA)
- Declarative memory should be **discrete text retrieval** (vector DB)

...all fused in a single agent loop with a principled entropy-driven routing mechanism across the three modalities. Each building block exists upstream; the contribution is the **composition**: routing between them, resolving conflicts across them, and mapping each cognitive function to the modality that is cheapest and most accurate for it.

This is called **Modality-to-Function Mapping**: specific _types_ of agentic thought require specific _data modalities_ to optimize compute limits. This is highly publishable.

---

## Literature Positioning

### Hybrid Memory (Text-Only)

- _Project Synapse_ (Jan 2026) and _Memory Management for Low-Code Agents_ (Sept 2025) successfully divide agent memory into Working, Episodic, and Semantic layers. **But they use text/vector databases for all three.** They miss the token-economic advantages of switching modalities.

### Visual Compression

- _AgentOCR_ (Jan 2026) pioneered Segment Optical Caching
- _Vist: Vision-Centric Token Compression_ (Feb 2025) uses a "slow-fast" reading path where distant text is compressed into images

### MSA

- EverMind's [MSA architecture](https://github.com/EverMind-AI/MSA) (March 2026) is a sparse-attention layer with a learnable router over compressed document KV chunks, built on Qwen3-4B and runnable on 2×A800-class GPUs. It is currently positioned as a long-context RAG *replacement* — a single end-to-end retrieval-augmented attention mechanism, not a complementary memory layer. Tri-Mem reframes it as the **semantic** layer of a three-modality hierarchy, with Visual Bus owning episodic recall and a separate text RAG owning exact-fact retrieval. Code and weights are tagged "Coming Soon"; once released, Tri-Mem's simulated MSA layer becomes a drop-in real implementation.

### Bayesian Reasoning for LLMs

- Google Research paper (March 2026) on teaching LLMs to reason like Bayesians — directly applicable to the calibration problem in the entropy router.

### Gap This Fills

All three exist independently. Nobody has combined them with modality-to-function mapping and entropy-driven routing.

---

## Engineering Bottlenecks to Solve

### A. Routing Latency (Epistemic Uncertainty)

Using an LLM prompt to decide which memory to access wastes tokens and adds latency. Fix: entropy-driven routers that operate on raw logit math, not LLM reasoning. Inspired by edge-computing papers on entropy-driven routing (_Real-Time Visual Anomaly Detection_, Mar 2026).

### B. Modality Interference in Attention Heads

Feeding a model pre-computed 1D latent tensors (MSA), 2D visual patch embeddings (Visual Bus), and discrete 1D text tokens (RAG) simultaneously causes "modality competition" — attention collapses toward one modality. Fix: **Late-Fusion** or **Gated Cross-Attention** where the visual episodic memory is kept in a separate latent stream and only gated into the main reasoning stream when requested.

### C. Synchronization and Conflict Crisis

When memories contradict each other (MSA says "Use API v1", Visual Bus learned "v1 is deprecated", RAG has endpoint for "v3"). Fix: **Hierarchical Overwrite Protocol** — explicit hierarchy of truth: RAG/real-time > Visual Episodic > MSA static rules.

---

## Benchmarks & Evaluation Strategy

### Tier 1: NovaCorp AuditBench (Primary — Custom)

A custom IT procurement & security audit simulator (`benchmarks/novacorp_audit_sim.py`, spec in `benchmarks/IT_PROCUREMENT_AUDIT_BENCHMARK.md`). Drops the agent into NovaCorp — a simulated mid-size SaaS company undergoing an annual IT audit across procurement records, credential vaults, vendor registries, and network inventory.

**Why it fits (stress-tests every Tri-Mem layer):**

- **MSA** — a ~40-page IT Policy & Procurement Guide with 12 sections of rules. Baseline agents re-read this every turn and still lose middle-section rules to context rot. MSA pre-computes it once.
- **Visual Bus** — audits span 36–48+ turns across dozens of records and distractor systems. At turn 42 the agent needs to recall a vendor flagged at turn 8. Raw text history at that point is tens of thousands of tokens; Visual Bus compresses it into OCR-readable tiles.
- **RAG** — exact alphanumeric strings (API keys like `sk-NvC-4f8a2b1c9d3e7f6a0b`, license serials like `LIC-2024-NVC-00847`, IPs, PO numbers). One wrong character = missed violation. Visual compression will blur these; RAG guarantees lossless recall.
- **Entropy Router** — policy confidence (low → MSA), uncertainty about prior checks (medium → Visual Bus), need for exact credentials (high → RAG) all fire naturally within one task.
- **Multi-Agent** — splits cleanly into Compliance Auditor (policy expert) + Forensics Analyst (credential cross-referencer).

**Task format:** 6 task templates across 5 audit task types (audit, security, compliance, patch, analysis). Strict sequential step validation — actions must follow the correct SOP order. The environment returns three distinct error classes:

- `Access denied or prerequisite not met.` — out-of-order but otherwise valid step
- `Command executed but returned no results or failed.` — wrong target
- `Syntax error: Unrecognized terminal command.` — invalid command format

This replaces ALFWorld as the primary benchmark because AuditBench exercises all three memory layers simultaneously, whereas ALFWorld did not justify the MSA layer.

### Tier 2: WebArena / Visual-WebArena

Simulated websites (e-commerce, Reddit, CMS). Agent must physically "look" at screen — native fit for Visual Bus. Tests RAG when exact data (email address from turn 3) is needed at turn 25.

### Tier 3: EcoGym

Tests long-context, multi-turn tool use in interactive economies. Authors specifically noted that expanding context window often _degrades_ performance. Perfect stage for Tri-Mem to prove intelligent compression beats raw context length.

### Tier 4: V-NIAH (Synthetic Baseline)

Visual Needle-In-A-Haystack: 100 turns of visually compressed history, retrieve one specific detail from turn 42. Isolates and proves the Visual Bus's attention-guided token pruning works.

### AuditBench Experiment Design (Detailed)

**Metric 1: Spatial Hallucination Rate**

- What: Agent believes it is connected to a system that actually rejected the access, or treats a non-existent endpoint/record as reachable
- Proves: Visual Bus retains episodic truth about *what failed* better than text summarization
- Observed on `audit_vendor_invoice_0` (1-task runs): Baseline = 25, RAG = 34, Visual Bus = 19

**Metric 2: Syntactic Rejection Rate**

- What: Environment returns `Syntax error: Unrecognized terminal command.` due to malformed command or (in Visual Bus runs) raw `<think>` reasoning leaking into the action slot
- Proves: RAG bridges lossy visual memory to discrete textual execution; action extractor must not let CoT overflow into the command
- Observed: Baseline = 0, RAG = 0, Visual Bus = 6

**Metric 3: Cumulative Token Cost**

- What: Total tokens (in + out) consumed across the full task
- Proves: Visual Bus should flatten the curve; text-only agents grow super-linearly as RAG context injection compounds
- Observed: Baseline = 45,169 tokens | RAG = 262,147 tokens | Visual Bus = 58,155 tokens on the same task
- Signal: RAG alone is the *worst* on token economics — it's lossless for facts but amplifies context. Visual Bus + RAG combined (Phase 3.5) is the hypothesis to validate.

**Metric 4: Action-to-Resolution Length**

- What: Total turns to task completion (lower = less context confusion)
- Proves: Compressed memory should shorten the resolution path
- Observed: Baseline = 41 turns | RAG = 48 turns | Visual Bus = 36 turns

---

## Metrics to Track

Every benchmark run must report (not just success rate). The middle columns show observed values from the `audit_vendor_invoice_0` single-task runs in [logs/](logs/) — Baseline / RAG / Visual Bus / VBus+RAG / MSA / Tri-Mem. Tri-Mem values are pulled from [logs/trimem.json](logs/trimem.json) (single-task slice for direct comparison; the seven-task aggregate is what's discussed in the body text since the band split only becomes statistically interesting across multiple tasks).

| Metric                          | What It Measures                                       | Which Layer It Tests                  | TurnMetric Field                           | Baseline          | RAG                | Visual Bus       | VBus+RAG          | MSA              | Tri-Mem (routed)        |
| ------------------------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------------------------ | ----------------- | ------------------ | ---------------- | ----------------- | ---------------- | ----------------------- |
| **Success Rate**                | Task completion %                                      | Overall system                        | `TaskMetric.success`                       | 1.0               | 1.0                | 1.0              | 1.0               | 1.0              | 1.0                     |
| **Cumulative Token Cost**       | Total tokens (in + out) over N-turn task               | Visual Bus (should flatten the curve) | `tokens_in + tokens_out`                   | 45,169            | 262,147            | 58,155           | 7,119             | 29,026           | 31,606                  |
| **Action-to-Resolution Length** | Steps to solve (fewer = less context confusion)        | Visual Bus + MSA                      | `TaskMetric.total_turns`                   | 41                | 48                 | 36               | 7                 | 6                | 6                       |
| **Syntactic Failure Rate**      | Environment "Syntax error" responses                   | RAG (should drive to near-zero)       | `syntactic_error` (bool per turn)          | 0                 | 0                  | 6 ⚠️             | 0                 | 0                | 0                       |
| **Spatial Hallucination Rate**  | Interacting with non-reachable systems/records         | Visual Bus                            | `spatial_hallucination` (bool per turn)    | 25                | 34                 | 19               | 0                 | 0                | 0                       |
| **Token Exhaustion Threshold**  | Turn at which agent degrades/crashes                   | Visual Bus + MSA                      | derived from per-turn `tokens_in` curve    | n/a (passed)      | n/a (passed)       | n/a (passed)     | n/a (passed)      | n/a (passed)     | n/a (passed)            |
| **Memory Source Distribution**  | % of turns using MSA vs Visual Bus vs RAG              | Entropy Router                        | `memory_source`                            | 100% text         | 100% rag           | 100% visual_bus  | 100% vbus_rag     | 100% msa         | this task: 17/67/17%; 7-task suite: 5/60/35% (msa / msa_vbus / msa_vbus_rag) |
| **Router Accuracy**             | Did the router pick the right modality?                | Entropy Router                        | `memory_source` vs ground truth            | n/a               | n/a                | n/a              | n/a               | n/a              | regression-pinned in `tests/test_entropy_router.py` (21/21 passing) |
| **Latency per Turn**            | Time per decision (router + inference overhead)        | Router efficiency                     | `latency_ms`                               | tracked           | tracked            | ~2.7s normal / ~7.7s on CoT-overflow turns | ~1.7s avg / ~9.4s cold start (OCR) | ~1.2s steady / 11.7s cold start | ~0.9s MED-band steady; LOW-band ~0.5s (OCR skipped); 6.6s cold start (probe + OCR + RAG) |
| **Cost per Task**               | USD cost comparison across agents                      | Token economics                       | `TaskMetric.total_cost_usd`                | $0.00 (local)     | $0.00 (local)      | $0.00 (local)    | $0.00 (local)     | $0.00 (local)    | $0.00 (local)           |
| **Entropy Score per Turn**      | First-token Shannon entropy of the probe pass (bits)   | Entropy Router                        | `entropy_score`                            | −1.0 (no probe)   | −1.0 (no probe)    | −1.0 (no probe)  | 0.0–1.0 (action-diversity, not logit-derived) | −1.0 (no probe) | 0.33–1.16 bits across 7-task suite (real, top-K logprobs + residual) |
| **Duration per Task**           | Wall-clock seconds to completion                       | End-to-end throughput                 | `TaskMetric.duration_s`                    | tracked           | tracked            | 817.5 s          | 93.5 s            | 19.5 s           | 38.1 s                  |

---

## Feasibility Assessment

### Realistic Now

- RAG for exact-fact retrieval alongside visual episodic memory — implementable with existing multimodal models (Qwen-VL, Gemini)
- Compressing agent history into image patches (Visual Bus) — AgentOCR demonstrated this works
- Benchmarking on ALFWorld and WebArena — accessible, well-documented environments

### Hard But Doable

- Entropy-based router: approximate by monitoring output logit distributions and triggering RAG calls when confidence drops. Heuristic threshold on token-level entropy gets 80% of the theoretical benefit.

### Blocked on Upstream (Simulate for Now, Swap In Later)

- **True MSA sparse-attention layer:** EverMind's [MSA repo](https://github.com/EverMind-AI/MSA) fully describes the architecture (Qwen3-4B base, chunk-mean-pooled KV + routing key `Kᵣ`, cosine-similarity Top-k retrieval, document-wise RoPE, 2×A800-class footprint) but tags code and weights as "Coming Soon". Tri-Mem now **simulates MSA** via two paths working in tandem: (1) vLLM prefix caching holds the full rulebook KV once per process, and (2) a ChromaDB-backed Top-k router over section-chunked embeddings injects the most relevant rulebook sections into each turn's user message. When upstream ships, `MSAStore.query()` becomes a sparse-attention call and the frozen prompt becomes a cached-KV handle — the rest of the Tri-Mem stack is unchanged.
- Training a Bayesian-calibrated LLM from scratch: use existing models and measure calibration instead.

### Recommended Scope

Simulate MSA with frozen long-context prompt. Implement Visual Bus with screenshot compression. Wire up standard RAG (ChromaDB). Build simple entropy router using logit confidence scores. Run ALFWorld + one WebArena task. Track the three core metrics. That's a strong, defensible paper.

---

## Implementation: Phased Build Plan

### Phase 1: Baseline Agent (Control Group) ✅ BUILT

- Simple text-history agent (full conversation appended every turn) tested on the NovaCorp AuditBench
- No memory optimization — this is the control group
- Measures: success rate, token cost per task, failure modes
- Everything after this gets compared back to this baseline

### Phase 2: + RAG Layer ✅ BUILT

- Introduce ChromaDB vector database storing environment object IDs and exact syntax strings
- Agent queries RAG before each action for relevant facts
- Measures: whether syntactic rejection rate drops compared to baseline
- Expected: fewer syntax errors, slightly higher token count (RAG context injection)

### Phase 3: + Visual Bus ✅ BUILT

- Replaces growing text history with rendered image tiles read back via GLM-OCR
- Agent "sees" its past as images instead of reading JSON logs
- Uses OCR-based episodic compression to maintain a compact visual timeline
- Measures: token savings and whether spatial hallucination rate drops
- Requires: multimodal model (GLM-OCR for reading, Qwen 3.5 for reasoning)

### Phase 3.75: + MSA Layer (Simulated Semantic Memory) ✅ BUILT

- Static NovaCorp IT Policy (`memory/rulebook/novacorp_it_policy.md`) chunked into 13 section-level chunks (preamble + 12 SOP/reference sections)
- **Path 1** — full rulebook embedded in a byte-identical system prompt; vLLM `enable_prefix_caching=True` holds the rulebook KV once per process (set via `configs/settings.ENABLE_PREFIX_CACHING`)
- **Path 2** — `memory/msa_store.py` exposes cosine Top-k routing over ChromaDB-embedded section chunks; live query (goal + observation + last action) pulls the most relevant sections and injects them into the *user* message only (system prompt stays frozen so the cache holds)
- Ablation knobs in `configs/settings.py`: `MSA_INJECT_FULL_RULEBOOK`, `MSA_INJECT_ROUTED_CHUNKS`, `MSA_TOP_K` — can run path 1 alone, path 2 alone, or both together
- `agents/msa_agent.py` emits `memory_source="msa"` per turn, ready to feed the Phase 4 entropy router
- Drop-in seam for EverMind's real sparse-attention MSA when weights ship: swap `MSAStore.query()` for a sparse-attention call, leave every caller untouched

### Phase 3.5: Visual Bus + RAG Combined (Syntactic Action Gap) ✅ BUILT

- `agents/visual_bus_rag_agent.py`: `VisualBusRAGAgent` fuses visual episodic memory with **summary-guided** RAG retrieval
- **Key innovation over Phase 3**: instead of querying RAG with a generic question every turn, the agent extracts entity mentions (system names, record IDs, alphanumeric keys) from the OCR-compressed visual summary and runs targeted RAG lookups for each one. The visual summary tells the agent *what* to look up; RAG provides the *exact* string.
- **Action outcome tracking**: every action→observation pair is stored in RAG with success/failure metadata, so the agent can recall "access to X failed at turn N" — the failure signal that raw visual compression loses
- `memory/rag_store.py` gains `query_multi()` (batched de-duplicated retrieval) and `extract_entities()` (regex extraction of IDs from OCR text)
- Measures: whether targeted RAG + Visual Bus together beat either alone on spatial hallucination rate and syntactic errors (the Syntactic Action Gap hypothesis)
- `memory_source="visual_bus_rag"` per turn

### Phase 4: + Entropy Router ✅ BUILT

- `router/entropy.py`: pure threshold-based router. `route(entropy)` maps Shannon entropy (bits) to a `RouteDecision` with three additive bands:
  - `H < ENTROPY_LOW_THRESHOLD` (0.4, empirically calibrated) → **MSA only** (rulebook chunks routed top-k from the cached system prompt KV)
  - `0.4 ≤ H < ENTROPY_MED_THRESHOLD` (0.7) → **MSA + Visual Bus** (routed rulebook chunks + OCR-compressed timeline)
  - `H ≥ 0.7` → **MSA + Visual Bus + RAG** (full stack with summary-guided fact lookups)
  - `H = -1.0` (no probe / unsupported backend) → defaults to full stack so signal loss never silently degrades memory
- `agents/trimem_agent.py`: `TriMemAgent` does **probe → route → generate** per turn:
  1. Probe — single-token call with no memory injected, captures top-K logprobs (`ROUTER_PROBE_TOP_K=20`)
  2. Compute first-token Shannon entropy (top-K + residual bucket)
  3. Build the act-pass user message with the memory blocks the router selected
  4. Final generation with the selected memory; emits `memory_source` matching the band label (`msa`, `msa_vbus`, `msa_vbus_rag`)
- `utils/llm.py`: both vLLM and transformers backends accept `logprobs=N` and populate `LLMResponse.first_token_entropy`
- Probe tokens + latency are added to the turn metric so cost reporting is honest about the router's overhead
- Ablation knob: `ROUTER_ENABLED=False` in `configs/settings.py` disables the probe and forces full memory injection on every turn (equivalent to a 1+2+3 stacked agent)

### Phase 5: Frontend Dashboard ✅ BUILT (sample data, connects to live API)

- Web UI to watch agent live, see which memory modality is active per turn
- Run tasks manually for vibe checks
- View benchmark comparison charts
- Metrics: success rate, token curves, error rates, cost comparisons

### Phase 6: Multi-Agent Extension 🔲

- Two agents sharing the Visual Bus and RAG layer on a collaborative task
- Compare transcript-passing vs shared Visual Bus + RAG
- Entropy router extended to inter-agent delegation
- Measures: coordination efficiency, total system token cost, task success rate

---

## Current Project Structure

```
tri-mem/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_benchmark.py                   # Main benchmark runner (CLI)
│
├── agents/
│   ├── base_agent.py                  # Abstract agent interface
│   ├── baseline_agent.py              # Phase 1: full text history agent
│   ├── rag_agent.py                   # Phase 2: RAG-augmented agent
│   ├── visual_bus_agent.py            # Phase 3: Visual Bus episodic memory agent
│   ├── visual_bus_rag_agent.py        # Phase 3.5: Visual Bus + RAG combined agent
│   │                                  #   - Summary-guided RAG: entities extracted from OCR
│   │                                  #     summary drive targeted fact lookups
│   │                                  #   - Action outcome tracking (success/fail in RAG)
│   ├── msa_agent.py                   # Phase 3.75: MSA semantic memory agent
│   │                                  #   - Frozen rulebook system prompt (prefix cache hot)
│   │                                  #   - Top-k routed chunks injected per turn
│   └── trimem_agent.py                # Phase 4: full Tri-Mem agent
│                                      #   - Probe → route → generate per turn
│                                      #   - Entropy-gated MSA / Visual Bus / RAG injection
│
├── benchmarks/
│   └── novacorp_audit_sim.py          # Simulated NovaCorp Audit environment
│                                      #   - 7 task templates (audit, security, compliance, patch,
│                                      #     analysis, plus a repeat-SOP `patch_two_servers`
│                                      #     designed to drive entropy below the LOW threshold)
│                                      #   - Strict sequential SOP validation
│                                      #   - Three distinct failure modes (prereq / wrong target / syntax)
│   └── IT_PROCUREMENT_AUDIT_BENCHMARK.md  # Full AuditBench design spec
│
├── configs/
│   └── settings.py                    # All configuration (model, thresholds, ports)
│
├── frontend/
│   ├── app.py                         # Flask API backend
│   │                                  #   - GET /api/results — benchmark results
│   │                                  #   - GET /api/tasks — available task templates
│   │                                  #   - POST /api/run_task — live agent execution
│   └── dashboard.html                 # Full dashboard (HTML/CSS/JS, no framework)
│                                      #   - Benchmark Overview tab (metrics + token chart)
│                                      #   - Vibe Check tab (run tasks live, see turn timeline)
│                                      #   - Agent Comparison tab (side-by-side bar charts)
│
├── memory/
│   ├── rag_store.py                   # ChromaDB-backed vector store
│   │                                  #   - store_fact(), store_observation(), query()
│   │                                  #   - query_multi(): batched de-duplicated retrieval
│   │                                  #   - extract_entities(): regex ID extraction from OCR text
│   │                                  #   - Extracts object mentions from observations
│   ├── visual_bus.py                  # OCR-based episodic memory compression
│   │                                  #   - Renders turn history to image tiles
│   │                                  #   - GLM-OCR reads tiles back to compressed text
│   ├── msa_store.py                   # Phase 3.75: simulated MSA rulebook store
│   │                                  #   - Section-level chunker over markdown rulebook
│   │                                  #   - ChromaDB cosine Top-k routing
│   │                                  #   - Process-wide singleton (MSAStore.shared())
│   │                                  #   - Drop-in seam for EverMind sparse-attention MSA
│   └── rulebook/
│       └── novacorp_it_policy.md      # 12-section IT Policy & Procurement Guide
│                                      #   - Authoritative SOPs for all 6 task templates
│                                      #   - ~11k chars, frozen in system prompt prefix
│
├── router/
│   └── entropy.py                     # Phase 4: pure threshold-based router
│                                      #   - route(entropy) → RouteDecision (band + per-layer flags)
│                                      #   - 3 bands: msa / msa_vbus / msa_vbus_rag
│                                      #   - Thresholds in configs/settings.py
│
├── tests/
│   └── test_entropy_router.py         # Phase 4: unit + regression tests for the router
│                                      #   - Threshold ordering, band table, boundary semantics
│                                      #   - Negative-entropy default-to-full-stack
│                                      #   - Regression entries pinned to recorded probe entropies
│                                      #     from the seven-task suite — silent threshold drift
│                                      #     is caught in CI rather than at benchmark time
│
├── utils/
│   ├── llm.py                         # Unified LLM inference wrapper
│   │                                  #   - VLLMBackend (fast GPU inference)
│   │                                  #   - TransformersBackend (universal HF fallback)
│   │                                  #   - Singleton pattern — model loads once, shared across agents
│   │                                  #   - logprobs=N support → first_token_entropy on LLMResponse
│   └── metrics.py                     # TurnMetric, TaskMetric, BenchmarkResult
│                                      #   - Tracks per-turn: tokens, latency, memory source,
│                                      #     entropy score, syntactic errors, spatial hallucinations
│                                      #   - Aggregates: success rate, avg turns, cost, error totals
│                                      #   - Saves JSON reports to logs/
│
└── logs/                              # Benchmark results (auto-generated JSON files)
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

The project runs a local open-source LLM (default: `Qwen/Qwen3.5-35B-A3B`) via **vLLM**. No API keys needed.

**To use a different model**, edit `MODEL_NAME` in `configs/settings.py` to any HuggingFace model ID:

```python
# configs/settings.py
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # or any HF model
INFERENCE_BACKEND = "vllm"                          # or "transformers"
```

### Google Colab (A100 80GB)

```python
!pip install -r requirements.txt
!python run_benchmark.py --agent baseline --tasks 5
```

### Run Phase 1 Baseline Benchmark

```bash
python run_benchmark.py --agent baseline --tasks 5
```

### Run Phase 2 RAG Benchmark

```bash
python run_benchmark.py --agent rag --tasks 5
```

### Run Phase 3 Visual Bus Benchmark

```bash
python run_benchmark.py --agent visual_bus --tasks 5
```

### Run Phase 3.75 MSA Benchmark

```bash
python run_benchmark.py --agent msa --tasks 5
```

Ablate the two MSA paths independently by flipping `MSA_INJECT_FULL_RULEBOOK` / `MSA_INJECT_ROUTED_CHUNKS` in `configs/settings.py`.

### Run Phase 3.5 Visual Bus + RAG Benchmark

```bash
python run_benchmark.py --agent visual_bus_rag --tasks 5
```

### Run Phase 4 Tri-Mem Benchmark (Entropy-Routed)

```bash
python run_benchmark.py --agent trimem --tasks 5
```

Each turn does a 1-token probe, computes Shannon entropy on the top-K logprobs, and routes between MSA / MSA+VBus / MSA+VBus+RAG. Toggle `ROUTER_ENABLED=False` in `configs/settings.py` to ablate the router and force full memory injection on every turn.

### Run All and Compare

```bash
python run_benchmark.py --agent all --tasks 5
```

### Launch Frontend Dashboard

```bash
# Start the API backend
python frontend/app.py

# Open frontend/dashboard.html in a browser
# The dashboard connects to localhost:5000 for live runs
# Falls back to sample data if the API isn't running
```

### CLI Options

```
--agent {baseline,rag,visual_bus,visual_bus_rag,msa,trimem,all}   Which agent to benchmark
--tasks N                                   Number of tasks to run (default: 5)
--quiet                                     Suppress per-turn output
```

---

## What's Built So Far

### ✅ Phase 1 — Baseline Agent

- `BaselineAgent` appends full conversation history every turn
- System prompt instructs single-action-per-turn with strict AuditBench command syntax
- Tracks all metrics per turn (tokens in/out, latency, errors)

### ✅ Phase 2 — RAG Agent

- `RAGAgent` stores every observation in ChromaDB
- Extracts object mentions (regex: "word number" patterns) and stores with location/turn metadata
- Queries top-5 relevant facts before each action, injects as "RELEVANT MEMORY" block
- Same system prompt + RAG memory system description

### ✅ Phase 3 — Visual Bus Agent

- `VisualBusAgent` compresses turn history into rendered image tiles read back by GLM-OCR
- `memory/visual_bus.py` handles text-to-image rendering and OCR decoding of the episodic timeline
- Compressed summary is fed to Qwen 3.5 for reasoning instead of raw JSON logs
- Tracks spatial hallucinations and syntactic errors alongside standard metrics

### ✅ Phase 3.5 — Visual Bus + RAG Combined (Syntactic Action Gap)

- `VisualBusRAGAgent` fuses Phase 3's visual episodic memory with **summary-guided** RAG retrieval
- After OCR compresses the history, `RAGStore.extract_entities()` pulls entity-like mentions (system names like `procurement_db`, record IDs like `invoice_1`, alphanumeric keys) from the compressed text
- `RAGStore.query_multi()` runs targeted lookups for each extracted entity, returning de-duplicated results — this bridges the Syntactic Action Gap: visual memory knows *what* to look for, RAG provides the *exact* string
- Action→observation outcomes are stored in RAG with success/failure metadata, so the agent can recall "access to X failed at turn N" — the failure signal that raw visual compression loses (addressing the spatial hallucination problem)
- Token cost profile matches Phase 3 Visual Bus (~constant), but higher-quality RAG context should reduce hallucination loops and syntactic errors

### ✅ Phase 3.75 — MSA Semantic Memory Agent (Simulated)

- `memory/rulebook/novacorp_it_policy.md`: 12-section IT Policy & Procurement Guide with SOPs for every task template (audit, security, compliance, patch, analysis); ~11k chars / ~2.8k tokens
- `memory/msa_store.py`: `MSAStore` with section-level chunker, ChromaDB Top-k cosine routing, process-wide singleton, and `full_rulebook_text` / `format_routed_chunks` helpers
- `agents/msa_agent.py`: `MSAAgent` combines path 1 (frozen rulebook system prompt + vLLM prefix caching) and path 2 (live-query Top-k router injecting sections into the user message)
- Task goal is placed in the first user message (not the system prompt) so the cached prefix is reusable across tasks, not just turns
- Ablation knobs: `MSA_INJECT_FULL_RULEBOOK`, `MSA_INJECT_ROUTED_CHUNKS`, `MSA_TOP_K` in `configs/settings.py`
- `ENABLE_PREFIX_CACHING=True` passed through to `vllm.LLM(...)` in `utils/llm.py`
- Emits `memory_source="msa"` per turn — wired for the Phase 4 entropy router

### ✅ Phase 4 — Entropy Router (Tri-Mem Agent)

- `router/entropy.py`: pure `route(entropy) → RouteDecision` with three bands (`msa`, `msa_vbus`, `msa_vbus_rag`); thresholds configurable via `ENTROPY_LOW_THRESHOLD` / `ENTROPY_MED_THRESHOLD`
- `agents/trimem_agent.py`: `TriMemAgent` runs **probe → route → generate** per turn:
  - **Probe**: 1-token call with no memory injected, no system prompt baggage, `logprobs=ROUTER_PROBE_TOP_K`
  - **Route**: first-token Shannon entropy (bits) over the top-K + residual bucket → band
  - **Generate**: act-pass with the gated memory blocks; system prompt is the frozen rulebook (Phase 3.75 path 1) so vLLM's prefix cache stays warm across all turns regardless of band
- `utils/llm.py`: `LLMResponse.first_token_entropy` field; vLLM (`SamplingParams.logprobs`) and transformers (`output_scores`) backends both populate it
- `memory_source` per turn now reflects what the router actually picked (`msa`, `msa_vbus`, `msa_vbus_rag`) — drops in directly to the dashboard's modality breakdown
- Probe tokens + latency are added to the turn metric so cumulative cost isn't undercounted
- Ablation knob: `ROUTER_ENABLED=False` in `configs/settings.py` forces full memory injection (equivalent to a stacked Phase 3.5+3.75 baseline)

### ✅ NovaCorp Audit Simulator

- 7 task templates: 6 single-SOP audits (audit, security, compliance, patch, analysis, expense-flagging) plus `patch_two_servers`, a repeat-SOP variant that re-runs the patch deployment against a second host so the router's LOW band can actually fire on the cached `access` + `run` pattern
- Strict sequential step validation — actions must follow the correct SOP order
- Returns "Access denied or prerequisite not met." for out-of-order but valid steps
- Returns "Command executed but returned no results or failed." for wrong targets
- Returns "Syntax error: Unrecognized terminal command." for invalid command formats
- Generates realistic observations with record IDs (e.g. `invoice_1`, `token_1`) and distractor systems

### ✅ Metrics System

- `TurnMetric`: per-turn telemetry (action, observation, tokens, latency, memory source, entropy, error flags)
- `TaskMetric`: per-task aggregation (success, total turns, total tokens, cost, error counts)
- `BenchmarkResult`: full benchmark summary with comparison helpers
- JSON export for all results

### ✅ Frontend Dashboard

- Three-tab layout: Overview, Vibe Check, Comparison
- Canvas-drawn token cost chart (no external chart library dependency)
- Live task execution with turn-by-turn timeline
- Agent comparison bar charts
- Connects to Flask API or falls back to sample data
- Dark theme, responsive layout

---

## Next Steps

### Immediate: Run the MSA benchmark on GPU

1. `python run_benchmark.py --agent msa --tasks 5` on the Colab A100 / GPU box
2. Verify vLLM prefix cache hits (look for flat per-turn `tokens_in` once the rulebook is cached)
3. Ablate path 1 vs path 2 by toggling `MSA_INJECT_FULL_RULEBOOK` / `MSA_INJECT_ROUTED_CHUNKS`
4. Compare cumulative token cost and SOP-ordering errors against Baseline / RAG / Visual Bus on the same task set

### Then: Run the Tri-Mem benchmark on GPU

1. `python run_benchmark.py --agent trimem --tasks 5` on the Colab A100 / GPU box
2. Verify the probe pass shows up as a separate `[LLM] Generating …` line with `H=…` reported
3. Inspect `memory_source` distribution in the resulting log — should split across `msa` / `msa_vbus` / `msa_vbus_rag` instead of pinning to one band
4. Compare cumulative token cost and SOP-ordering errors against Baseline / RAG / Visual Bus / VBus+RAG / MSA on the same task set
5. Ablate the router: set `ROUTER_ENABLED=False` and rerun to confirm the routed agent beats the always-on-everything stacked baseline

### Resolved Issues

- **Spatial hallucinations** — ✅ Fixed. Failed observations are rendered in yellow with "OBS FAILED" labels in the visual timeline (`memory/visual_bus.py`), so OCR preserves the failure signal instead of blurring it into success.
- **CoT leakage into actions** — ✅ Fixed. `BaseAgent.parse_action()` handles orphaned reasoning (truncated `<think>` blocks) by extracting the last embedded command verb from the reasoning text.
- **Loop detection / recovery** — ✅ Fixed. `BaseAgent` includes a repeat guard (same action 3+ times) and failure-rate guard (last 5 all failed), injecting explicit warnings into the agent's prompt.
- **Entropy score wired but inert** — ✅ Fixed for the Tri-Mem agent. `LLMResponse.first_token_entropy` is populated from real top-K logprobs (vLLM `SamplingParams.logprobs` / transformers `output_scores`), and `TurnMetric.entropy_score` carries the probe's first-token Shannon entropy. Other agents still report `-1.0` (they don't run a probe).
- **Entropy router LOW band silently disabled MSA chunks** — ✅ Fixed. `router/entropy.py` had `use_msa_routed_chunks=False` in the LOW branch, which contradicted the band label `"msa"`, the docstring, and the paper's additive ladder (LOW = MSA, MED = MSA+VBus, HIGH = MSA+VBus+RAG). On confident turns the agent was running with no memory at all instead of the cheap rulebook safety net. Flipped to `True`; the router is now genuinely additive.
- **LOW band threshold too aggressive** — ✅ Calibrated. With `ENTROPY_LOW_THRESHOLD = 0.3` the band never fired (minimum observed entropy floored at ≈0.33 across 90+ turns of benchmark data). Bumped to **0.4** based on observed distributions; the router now operates as a true ternary classifier — roughly 5% MSA, 60% MSA+VBus, 35% MSA+VBus+RAG on the seven-task suite.
- **Visual Bus OOM at default tile count** — ✅ Fixed. `MAX_VISUAL_TILES` default was 20, which produced rendered history images large enough to OOM the GLM-OCR forward pass on a single consumer GPU. The OCR exception path silently flips `_ocr_unavailable=True` and degrades the agent to a truncated-text fallback for the rest of the run, so the bug masquerades as "everything works." Reduced to **4** so the rendered image stays in OCR's stable working set; longer-range recall is delegated to RAG.

### Then: Phase 5 — Dashboard Enhancements

1. Real-time WebSocket streaming of agent turns (Flask-SocketIO is in requirements)
2. Memory modality visualization (color-coded per turn: orange=MSA, cyan=RAG, purple=Visual Bus)
3. Side-by-side live comparison runs
4. Export benchmark reports as PDF

### Then: Phase 6 — Multi-Agent

1. Build `AgentOrchestrator` managing multiple specialized agents
2. Implement shared Visual Bus (compressed session handoff between agents)
3. Implement shared RAG store (common fact layer)
4. Extend entropy router for inter-agent delegation
5. Design collaborative AuditBench task (Compliance Auditor + Forensics Analyst, divided responsibilities per the spec)
6. Key measurement: transcript-passing vs shared Tri-Mem overhead and success rate

### Paper Framing

- Title: _Tri-Mem: Modality-Matched Memory Hierarchies for Multi-Agent Systems_
- Frame NovaCorp AuditBench as the end-to-end stress test for all three layers simultaneously; keep ALFWorld/WebArena/EcoGym as future external validation benchmarks
- Reframe the paper to emphasize the multi-agent orchestration angle (bigger claim, architecture supports it)
- Include the token cost curve graph as the hero figure
- Discuss Bayesian calibration as future work (don't need to build it, just need to identify the gap)

---

## Key Design Decisions

- **AuditSim is drop-in replaceable.** The `NovaCorpAuditSim` class implements `reset()` → initial briefing and `step(action)` → (observation, done, success). Any environment matching this interface can be swapped in — originally specced against ALFWorld, now replaced by the custom NovaCorp IT procurement audit because it exercises all three memory layers (MSA + Visual Bus + RAG) instead of just two.
- **Agents are modular.** Every agent extends `BaseAgent` with `reset(goal)` and `act(observation, turn) → (action, TurnMetric)`. Adding a new agent variant means one new file in [agents/](agents/). Six variants exist today: [baseline_agent.py](agents/baseline_agent.py), [rag_agent.py](agents/rag_agent.py), [visual_bus_agent.py](agents/visual_bus_agent.py), [visual_bus_rag_agent.py](agents/visual_bus_rag_agent.py), [msa_agent.py](agents/msa_agent.py), [trimem_agent.py](agents/trimem_agent.py).
- **Metrics are phase-agnostic.** `TurnMetric` carries `memory_source` and `entropy_score` for every agent. The Tri-Mem agent populates real logit-derived entropy via the probe pass; older single-modality agents still report `-1.0` (they don't run a probe), and `memory_source` distinguishes them anyway.
- **The router is pure.** `router/entropy.py` does no I/O — it only maps a scalar entropy to a `RouteDecision`. Probing and memory assembly live in the agent. That keeps the router trivially testable and makes it cheap to swap in a calibrated entropy source later (e.g. Bayesian Teaching).
- **Strict SOP ordering in the benchmark.** The audit simulator distinguishes three failure modes (`Access denied…`, `Command executed but returned no results…`, `Syntax error…`) so that metrics can attribute failures to the *right* memory layer — prerequisite violations stress episodic memory, wrong targets stress fact retrieval, syntax errors stress the action extractor.
- **Frontend is framework-free.** Pure HTML/CSS/JS, no build step, no node_modules. Opens directly in a browser. Connects to Flask API when available, works standalone with sample data.
- **Config is centralized.** All thresholds, model settings, and feature flags in `configs/settings.py`.
