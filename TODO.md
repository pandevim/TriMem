1. Semantic Memory / MSA (Multi-Scale Attention) — OVERSTATED
   Paper claims (line 72): KV chunks are "pre-computed into K and V tensors alongside a routing key K_r," cosine-similarity matched, streamed to GPU, with "Document-wise RoPE."

Reality: No sparse attention, no KV tensors, no RoPE. memory/msa_store.py chunks the rulebook by ## headings and embeds them into ChromaDB with all-MiniLM-L6-v2. Routing is plain cosine similarity over these static embeddings — standard vector search, not a learned neural router. The MSA_V2_REAL_ATTENTION.md design doc explicitly calls the current approach a simulation and defers real MSA to "when EverMind ships." The README also states: "Tri-Mem currently simulates the layer with a two-path approach."

Additionally, configs/settings.py has MSA_INJECT_FULL_RULEBOOK = True, meaning the full rulebook text is injected into the system prompt and relies on vLLM prefix caching — not latent tensor attention.

2. Entropy Router — NOT FUNCTIONAL
   Paper claims (lines 81–90): An uncertainty-guided router monitoring "raw output logit confidence" using Shannon Entropy, with thresholds (<0.3, 0.3–0.7, >0.7) that dynamically route to MSA, Visual Bus, or RAG.

Reality: The entropy computation in agents/visual_bus_rag_agent.py calculates entropy over past action counts (a diversity measure), not over model logits. Every single log in the project shows entropy_score: -1.0 (the "not measured" default from utils/metrics.py). There is no routing decision logic anywhere — agents are independent implementations (BaselineAgent, RAGAgent, VisualBusAgent, MSAAgent, VisualBusRAGAgent), not a unified agent that switches modalities. The thresholds in configs/settings.py (ENTROPY_LOW_THRESHOLD, ENTROPY_MED_THRESHOLD) are defined but never consumed.

3. Visual Bus (Episodic Working Memory) — PARTIALLY TRUE
   Paper claims (lines 74–76): "Observation-action deltas are rendered into highly compressed image patches using Segment Optical Caching. The multimodal LLM processes its own timeline as a visual sequence."

Reality: memory/visual_bus.py does render history into PIL images with color-coding (blue/yellow/red). It runs GLM-OCR (0.9B) to extract text back. However, the LLM never sees the images — it receives the OCR'd text summary as [COMPRESSED HISTORY]\n{compressed}. So it's text-to-image-to-text, not "processing a visual sequence." The compression is real and effective (~84% token reduction), but the mechanism is different from what the paper describes.

4. Declarative Memory / RAG — TRUE
   Paper claims (lines 77–79): RAG with a vector database for exact string retrieval.

Reality: memory/rag_store.py uses ChromaDB with cosine similarity, entity extraction via regex, and query_multi() for targeted retrieval. This matches the paper's description. The only caveat: it's in-memory only (no persistence), but that's not a factual error in the paper.

5. Benchmark Results (Table I) — PARTIALLY SUPPORTED
   Paper claims (lines 117–133): 5 phases evaluated on NovaCorp AuditBench with specific numbers.

Claim Verified?
Phase 1 Baseline: 45,169 tokens, 41 turns, 25 spatial hallucinations Matches logs
Phase 2 RAG: 262,147 tokens, 48 turns, 34 spatial hallucinations No logs found to confirm
Phase 3 Visual Bus: 58,155 tokens, 36 turns, 6 syntactic failures, 19 spatial No logs found to confirm
Phase 4 VBus+RAG: 7,119 tokens, 7 turns, 0 failures Matches logs
Phase 5 MSA: 29,026 tokens, 6 turns, 0 failures Matches logs
Only 1 task ("Vendor Invoice Audit") was run per agent. The paper doesn't explicitly claim statistical significance, but presenting a single-run result in a conference paper table without noting n=1 is misleading.

8. Multi-Agent Orchestration (Section IV) — NOT IMPLEMENTED
   Paper claims (lines 92–94): Visual Bus becomes a "shared episodic medium," RAG serves as "single source of truth," Entropy Router handles "inter-agent delegation."

Reality: There is zero multi-agent code. No orchestrator, no shared memory, no delegation logic. The benchmarks/ directory describes 26 planted violations intended for multi-agent detection, but only single-agent runs against 1 simple task exist. The README marks "Phase 5 — Multi-Agent Extension" with an unchecked checkbox.

9. "Extends reliable agent horizons from ~30 turns to over 100 turns" (line 54) — UNSUPPORTED
   No experiment ran beyond 48 turns. The longest run was Phase 2 (RAG) at 48 turns. There is no evidence of 100+ turn runs anywhere in the codebase or logs.
