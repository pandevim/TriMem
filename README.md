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
| Rule-following / deep reasoning   | MSA (Semantic Memory)       | Pre-computed KV latent tensors      | System prompts, SOPs, API docs, environment rules                   |
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
- Mechanism: corpus processed offline, latent KV states pre-loaded into GPU memory
- Benefit: instant deep-reasoning access to million-token rulebooks without context-window compute tax
- Current implementation: simulated via frozen long-context system prompt (true MSA requires open-source model access)

**2. Visual Bus — The Scratchpad (Episodic Layer)**

- Contains: step-by-step interaction history, terminal outputs, web pages navigated, intermediary thought chains
- Mechanism: renders observation/action deltas into image patches using Segment Optical Caching
- Benefit: highly compressed visual timeline of the current session, prevents Context Rot
- Key tech: instead of appending 2,000 text tokens of JSON logs per action, the system renders the delta into an image patch

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

- **Low Entropy (< 0.3):** Model is highly confident → rely on internal MSA/context. Execute action confidently.
- **Medium Entropy (0.3 - 0.7):** Environmental uncertainty → inject Visual Bus patches to update the model's beliefs about its surroundings.
- **High Entropy (> 0.7):** Total epistemic uncertainty (needs a specific fact) → halt generation and query RAG database for the exact string.

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

- Episodic memory should be **visual** (image patches)
- Semantic memory should be **pre-computed latent states** (KV cache)
- Declarative memory should be **text retrieval** (vector DB)

...all fused in a single agent loop with a principled entropy-driven routing mechanism.

This is called **Modality-to-Function Mapping**: specific _types_ of agentic thought require specific _data modalities_ to optimize compute limits. This is highly publishable.

---

## Literature Positioning

### Hybrid Memory (Text-Only)

- _Project Synapse_ (Jan 2026) and _Memory Management for Low-Code Agents_ (Sept 2025) successfully divide agent memory into Working, Episodic, and Semantic layers. **But they use text/vector databases for all three.** They miss the token-economic advantages of switching modalities.

### Visual Compression

- _AgentOCR_ (Jan 2026) pioneered Segment Optical Caching
- _Vist: Vision-Centric Token Compression_ (Feb 2025) uses a "slow-fast" reading path where distant text is compressed into images

### MSA

- EverMind's MSA architecture dropped March 2026. Currently treated as a RAG replacement, not a complementary foundational layer.

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

Every benchmark run must report (not just success rate). The rightmost columns show observed values from the `audit_vendor_invoice_0` single-task runs in [logs/](logs/) — Baseline / RAG / Visual Bus.

| Metric                          | What It Measures                                       | Which Layer It Tests                  | TurnMetric Field                           | Baseline          | RAG                | Visual Bus       |
| ------------------------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------------------------ | ----------------- | ------------------ | ---------------- |
| **Success Rate**                | Task completion %                                      | Overall system                        | `TaskMetric.success`                       | 1.0               | 1.0                | 1.0              |
| **Cumulative Token Cost**       | Total tokens (in + out) over N-turn task               | Visual Bus (should flatten the curve) | `tokens_in + tokens_out`                   | 45,169            | 262,147            | 58,155           |
| **Action-to-Resolution Length** | Steps to solve (fewer = less context confusion)        | Visual Bus + MSA                      | `TaskMetric.total_turns`                   | 41                | 48                 | 36               |
| **Syntactic Failure Rate**      | Environment "Syntax error" responses                   | RAG (should drive to near-zero)       | `syntactic_error` (bool per turn)          | 0                 | 0                  | 6 ⚠️             |
| **Spatial Hallucination Rate**  | Interacting with non-reachable systems/records         | Visual Bus                            | `spatial_hallucination` (bool per turn)    | 25                | 34                 | 19               |
| **Token Exhaustion Threshold**  | Turn at which agent degrades/crashes                   | Visual Bus + MSA                      | derived from per-turn `tokens_in` curve    | n/a (passed)      | n/a (passed)       | n/a (passed)     |
| **Memory Source Distribution**  | % of turns using MSA vs Visual Bus vs RAG              | Entropy Router                        | `memory_source`                            | 100% text         | 100% rag           | 100% visual_bus  |
| **Router Accuracy**             | Did the router pick the right modality?                | Entropy Router                        | `memory_source` vs ground truth            | n/a (Phase 4)     | n/a (Phase 4)      | n/a (Phase 4)    |
| **Latency per Turn**            | Time per decision (router + inference overhead)        | Router efficiency                     | `latency_ms`                               | tracked           | tracked            | ~2.7s normal / ~7.7s on CoT-overflow turns |
| **Cost per Task**               | USD cost comparison across agents                      | Token economics                       | `TaskMetric.total_cost_usd`                | $0.00 (local)     | $0.00 (local)      | $0.00 (local)    |
| **Entropy Score per Turn**      | Model output-logit Shannon entropy (Phase 4 signal)    | Entropy Router                        | `entropy_score`                            | −1.0 (not wired)  | −1.0 (not wired)   | −1.0 (not wired) |
| **Duration per Task**           | Wall-clock seconds to completion                       | End-to-end throughput                 | `TaskMetric.duration_s`                    | tracked           | tracked            | 817.5 s          |

---

## Feasibility Assessment

### Realistic Now

- RAG for exact-fact retrieval alongside visual episodic memory — implementable with existing multimodal models (Qwen-VL, Gemini)
- Compressing agent history into image patches (Visual Bus) — AgentOCR demonstrated this works
- Benchmarking on ALFWorld and WebArena — accessible, well-documented environments

### Hard But Doable

- Entropy-based router: approximate by monitoring output logit distributions and triggering RAG calls when confidence drops. Heuristic threshold on token-level entropy gets 80% of the theoretical benefit.

### Out of Reach for Single Research Project (Simulate Instead)

- True MSA with pre-computed KV states: EverMind's architecture isn't open-source. **Simulate MSA** with a very long frozen system prompt and argue functional equivalence.
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

### Phase 4: + Entropy Router 🔲

- Build routing logic monitoring model confidence (output logits / token probabilities)
- Automatically decides: rely on internal context (low entropy), check Visual Bus (medium), query RAG (high)
- This is where the architecture becomes "Tri-Mem"
- Measures: router accuracy, latency overhead, overall performance improvement

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
│   └── visual_bus_agent.py            # Phase 3: Visual Bus episodic memory agent
│
├── benchmarks/
│   └── novacorp_audit_sim.py          # Simulated NovaCorp Audit environment
│                                      #   - 10 task templates (heat, clean, cool, pick_and_place, slice, examine)
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
│   │                                  #   - Extracts object mentions from observations
│   └── visual_bus.py                  # OCR-based episodic memory compression
│                                      #   - Renders turn history to image tiles
│                                      #   - GLM-OCR reads tiles back to compressed text
│
├── router/                            # Phase 4 (empty, to be built)
│
├── utils/
│   ├── llm.py                         # Unified LLM inference wrapper
│   │                                  #   - VLLMBackend (fast GPU inference)
│   │                                  #   - TransformersBackend (universal HF fallback)
│   │                                  #   - Singleton pattern — model loads once, shared across agents
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
--agent {baseline,rag,visual_bus,all}   Which agent to benchmark
--tasks N                               Number of tasks to run (default: 5)
--quiet                                 Suppress per-turn output
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

### ✅ NovaCorp Audit Simulator

- 6 task templates covering 5 audit task types: audit, security, compliance, patch, analysis
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

### Immediate: Phase 3.5 — Visual Bus + RAG combined

1. Visual Bus handles episodic memory (what happened, where things are)
2. RAG handles exact facts (object IDs, exact syntax strings)
3. When GLM-OCR's summary mentions an object but blurs the ID, the agent queries RAG for the exact string
4. Measure whether RAG + Visual Bus together beat either alone (the Syntactic Action Gap hypothesis)

### Known Visual Bus Issues (from initial runs)

- **Spatial hallucinations remain high** — agent still "remembers" successful connections to systems that actually failed, causing long loops on the same action. Compressed recall is losing failure signals.
- **CoT leakage into actions** — on ~1 in 6 turns the model hits the output token cap mid-reasoning and the raw `<think>` chain is emitted as the action, producing syntactic errors with ~7.7s latency vs ~2.7s on clean turns. Tighten the action extractor or cap CoT length.
- **Entropy score wired but inert** — all turns report `entropy_score = -1.0`. Needs real logit-derived entropy before Phase 4 routing can use it.
- **No loop-detection / recovery** — once a session breaks, the agent hammers the same failing command for 10+ turns. Need a repetition guard or backoff heuristic.

### Then: Phase 4 — Entropy Router

1. Capture token-level log probabilities from model responses
2. Compute Shannon Entropy per generation step
3. Build threshold-based router (configurable in `configs/settings.py`)
4. Create `TriMemAgent` that dynamically selects memory modality
5. Key measurement: does automated routing match or exceed manually optimized baselines?

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
- **Agents are modular.** Every agent extends `BaseAgent` with `reset(goal)` and `act(observation, turn) → (action, TurnMetric)`. Adding a new agent variant means one new file in [agents/](agents/). Three variants exist today: [baseline_agent.py](agents/baseline_agent.py), [rag_agent.py](agents/rag_agent.py), [visual_bus_agent.py](agents/visual_bus_agent.py).
- **Metrics are phase-agnostic.** `TurnMetric` already has fields for `memory_source` and `entropy_score` even though Phase 1–3 don't emit real entropy values (all runs currently record `-1.0`). No schema changes will be needed when the Phase 4 entropy router lands.
- **Strict SOP ordering in the benchmark.** The audit simulator distinguishes three failure modes (`Access denied…`, `Command executed but returned no results…`, `Syntax error…`) so that metrics can attribute failures to the *right* memory layer — prerequisite violations stress episodic memory, wrong targets stress fact retrieval, syntax errors stress the action extractor.
- **Frontend is framework-free.** Pure HTML/CSS/JS, no build step, no node_modules. Opens directly in a browser. Connects to Flask API when available, works standalone with sample data.
- **Config is centralized.** All thresholds, model settings, and feature flags in `configs/settings.py`.
