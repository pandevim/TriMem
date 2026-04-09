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

### Tier 1: ALFWorld (Primary — Phase III)

Drops an agent into a simulated household with a goal (e.g., "Wash an apple and put it in the fridge").

**Why it fits:**

- "Wandering" Forgetting Curve tests the Visual Bus — agent opens cabinet 3 on Turn 2, needs that info on Turn 45
- Strict Syntax Penalty tests RAG — environment rejects `put apple in fridge` but accepts `put apple 1 in/on fridge 1`

**Limitation (defend this):** ALFWorld doesn't justify the MSA layer (kitchen rules are simple). Frame it as the "Micro-Dynamics Benchmark" — isolating Visual Bus + RAG interplay, not testing MSA's upper bounds.

### Tier 2: WebArena / Visual-WebArena

Simulated websites (e-commerce, Reddit, CMS). Agent must physically "look" at screen — native fit for Visual Bus. Tests RAG when exact data (email address from turn 3) is needed at turn 25.

### Tier 3: EcoGym

Tests long-context, multi-turn tool use in interactive economies. Authors specifically noted that expanding context window often _degrades_ performance. Perfect stage for Tri-Mem to prove intelligent compression beats raw context length.

### Tier 4: V-NIAH (Synthetic Baseline)

Visual Needle-In-A-Haystack: 100 turns of visually compressed history, retrieve one specific detail from turn 42. Isolates and proves the Visual Bus's attention-guided token pruning works.

### ALFWorld Experiment Design (Detailed)

**Metric 1: Spatial Hallucination Rate**

- What: Agent tries to interact with object not in room, or searches empty cabinet again
- Proves: Visual Bus retains episodic history better than text summarization

**Metric 2: Syntactic Rejection Rate**

- What: Environment returns "Nothing happens." due to wrong command format or forgotten object ID
- Proves: RAG bridges lossy visual memory and discrete textual execution

**Metric 3: Token Exhaustion Threshold**

- What: Standard agents crash around turn 50 on API context limits
- Proves: Graph showing baseline token count scaling linearly (failing), Tri-Mem curving and plateauing (surviving to turn 100+)

---

## Metrics to Track

Every benchmark run must report (not just success rate):

| Metric                          | What It Measures                                | Which Layer It Tests                  |
| ------------------------------- | ----------------------------------------------- | ------------------------------------- |
| **Success Rate**                | Task completion %                               | Overall system                        |
| **Cumulative Token Cost**       | Total API cost over N-turn task                 | Visual Bus (should flatten the curve) |
| **Action-to-Resolution Length** | Steps to solve (fewer = less context confusion) | Visual Bus + MSA                      |
| **Syntactic Failure Rate**      | "Nothing happens" errors from wrong syntax      | RAG (should drive to near-zero)       |
| **Spatial Hallucination Rate**  | Interacting with wrong/nonexistent objects      | Visual Bus                            |
| **Token Exhaustion Threshold**  | Turn at which agent degrades/crashes            | Visual Bus + MSA                      |
| **Memory Source Distribution**  | % of turns using MSA vs Visual Bus vs RAG       | Entropy Router                        |
| **Router Accuracy**             | Did the router pick the right modality?         | Entropy Router                        |
| **Latency per Turn**            | Time per decision (router overhead)             | Router efficiency                     |
| **Cost per Task**               | USD cost comparison across agents               | Token economics                       |

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

- Simple ALFWorld agent using standard LLM with full text history appended every turn
- No memory optimization — this is the control group
- Measures: success rate, token cost per task, failure modes
- Everything after this gets compared back to this baseline

### Phase 2: + RAG Layer ✅ BUILT

- Introduce ChromaDB vector database storing environment object IDs and exact syntax strings
- Agent queries RAG before each action for relevant facts
- Measures: whether syntactic rejection rate drops compared to baseline
- Expected: fewer syntax errors, slightly higher token count (RAG context injection)

### Phase 3: + Visual Bus 🔲 NEXT

- Replace growing text history with compressed screenshot/image tiles
- Agent "sees" its past as images instead of reading JSON logs
- Uses Segment Optical Caching to maintain compressed visual timeline
- Measures: token savings and whether spatial hallucination rate drops
- Requires: multimodal model (Gemini, Qwen-VL, or Claude with vision)

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
│   └── rag_agent.py                   # Phase 2: RAG-augmented agent
│
├── benchmarks/
│   └── alfworld_sim.py                # Simulated ALFWorld environment
│                                      #   - 10 task templates (heat, clean, cool, pick_and_place, slice, examine)
│                                      #   - Strict syntax parsing (mirrors real ALFWorld)
│                                      #   - Drop-in replaceable with real ALFWorld
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
│   └── rag_store.py                   # ChromaDB-backed vector store
│                                      #   - store_fact(), store_observation(), query()
│                                      #   - Extracts object mentions from observations
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

### Run Both and Compare

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
--agent {baseline,rag,all}   Which agent to benchmark
--tasks N                    Number of tasks to run (default: 5)
--quiet                      Suppress per-turn output
```

---

## What's Built So Far

### ✅ Phase 1 — Baseline Agent

- `BaselineAgent` appends full conversation history every turn
- System prompt instructs single-action-per-turn with strict ALFWorld syntax
- Tracks all metrics per turn (tokens in/out, latency, errors)

### ✅ Phase 2 — RAG Agent

- `RAGAgent` stores every observation in ChromaDB
- Extracts object mentions (regex: "word number" patterns) and stores with location/turn metadata
- Queries top-5 relevant facts before each action, injects as "RELEVANT MEMORY" block
- Same system prompt + RAG memory system description

### ✅ ALFWorld Simulator

- 10 task templates covering 6 task types: heat, clean, cool, pick_and_place, slice, examine
- Strict sequential step validation (mirrors ALFWorld's rigid parser)
- Returns "Nothing happens." for wrong syntax or wrong order
- Returns "I don't understand that command." for invalid command formats
- Generates realistic observations with object IDs and distractor items

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

### Immediate: Phase 3 — Visual Bus

1. Implement screenshot/text-to-image rendering for agent observations
2. Build the Segment Optical Caching tile manager (compress, store, retrieve image patches)
3. Create `VisualBusAgent` that uses vision model to read compressed history
4. Requires switching to a multimodal open-source model (e.g., Qwen-VL)
5. Key measurement: does the token cost curve flatten while maintaining success rate?

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
5. Design collaborative ALFWorld task (two agents, divided responsibilities)
6. Key measurement: transcript-passing vs shared Tri-Mem overhead and success rate

### Paper Framing

- Title: _Tri-Mem: Modality-Matched Memory Hierarchies for Multi-Agent Systems_
- Frame ALFWorld as "Micro-Dynamics Benchmark" (tests Visual Bus + RAG interplay)
- Reframe the paper to emphasize the multi-agent orchestration angle (bigger claim, architecture supports it)
- Include the token cost curve graph as the hero figure
- Discuss Bayesian calibration as future work (don't need to build it, just need to identify the gap)

---

## Key Design Decisions

- **ALFWorld Sim is drop-in replaceable.** The `ALFWorldSim` class implements `reset()` → initial observation and `step(action)` → (observation, done, success). Swap in real ALFWorld by matching this interface.
- **Agents are modular.** Every agent extends `BaseAgent` with `reset(goal)` and `act(observation, turn) → (action, TurnMetric)`. Adding a new agent variant means one new file.
- **Metrics are phase-agnostic.** `TurnMetric` already has fields for `memory_source` and `entropy_score` even though Phase 1-2 don't use them. No schema changes needed as phases are added.
- **Frontend is framework-free.** Pure HTML/CSS/JS, no build step, no node_modules. Opens directly in a browser. Connects to Flask API when available, works standalone with sample data.
- **Config is centralized.** All thresholds, model settings, and feature flags in `configs/settings.py`.
