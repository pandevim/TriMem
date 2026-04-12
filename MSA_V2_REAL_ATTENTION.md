# MSA v2: Real Sparse-Attention Rulebook Layer

_Design doc for replacing the simulated MSA with a trained attention-level implementation._

---

## How This Differs from the Current MSA (v1, Phase 3.75)

The MSA we have today is a **text-level simulation**. It proves the routing concept and delivers prefix-cache savings, but the rulebook still lives in the context window as plain tokens:

| | MSA v1 (current, simulated) | MSA v2 (this doc, real attention) |
|---|---|---|
| **Where the rulebook lives** | In the system prompt as ~2.8k text tokens. vLLM prefix caching avoids re-computing the KV, but the tokens still occupy context window slots and compete for attention. | Pre-computed KV states loaded into GPU memory at startup. Zero tokens in the context window — the rulebook is latent, accessed only through modified attention heads. |
| **Routing mechanism** | ChromaDB cosine similarity over mean-pooled section embeddings. External to the model — runs before inference, injects Top-k chunk *text* into the user message. | Learned projection layers inside the upper attention heads. The model itself decides which chunks to attend to during the forward pass — end-to-end differentiable, no external retrieval step. |
| **What gets injected** | Full chunk text (~500-1k tokens per routed section) appended to the user message. The model reads it as regular tokens. | Compressed KV pairs (chunk-mean-pooled, ~1 KV pair per 128 source tokens) concatenated with the local KV before attention. The model attends over them, never "reads" them as text. |
| **Training** | None. Uses off-the-shelf ChromaDB default embeddings (all-MiniLM-L6-v2). | Router projection layers are fine-tuned on query→chunk relevance pairs (~500-1000 examples, few million parameters, single GPU). Base model weights stay frozen. |
| **Token cost** | System prompt is ~2.8k tokens every turn (prefix-cached so KV isn't recomputed, but still present in the attention window). Routed chunks add ~1-2k more tokens to the user message. | System prompt drops to 0 tokens. Routed chunks are KV states, not tokens — they add attention compute but not context length. |
| **vLLM compatibility** | Full compatibility. Standard prompt, standard inference. | Requires custom model class or fallback to transformers backend. vLLM's paged-attention kernel doesn't natively support injected external KV states. |
| **Drop-in interface** | `MSAStore.query(q) → list[RulebookChunk]` returning text. | Same `MSAStore.query(q)` interface, but returns pre-computed KV tensors instead of text. Every caller (agent, router, benchmark) stays unchanged. |

**In short:** v1 proves the *routing value* (correct SOP chunk reaches the model at the right turn) and the *caching value* (rulebook KV computed once). v2 proves the *attention-level mechanism* (rulebook removed from context entirely, accessed through learned sparse attention).

---

## Architecture

### Step 1: Modify the attention layer

Take Qwen 0.5B (prototype) or 1.5B (target). Add a routing step to the upper attention layers (top 8 out of 24). Before standard self-attention, the layer projects the current hidden state to a routing query, scores it against stored routing keys, fetches the Top-k chunks' KV states, and concatenates them with the local KV for one combined attention pass.

```python
class MSAAttentionLayer(nn.Module):
    def __init__(self, original_layer, num_routing_keys):
        self.original_layer = original_layer
        self.router_proj = nn.Linear(hidden_dim, routing_dim)  # small projection

    def forward(self, hidden_states, memory_keys, memory_values, memory_routing_keys):
        # Step 1: Project current hidden state to routing query
        routing_query = self.router_proj(hidden_states)  # [batch, seq, routing_dim]

        # Step 2: Score against stored routing keys
        scores = cosine_similarity(routing_query, memory_routing_keys)  # [batch, seq, num_chunks]

        # Step 3: Pick top-k chunks
        top_k_indices = scores.topk(k=4).indices

        # Step 4: Fetch those chunks' KV states
        selected_K = memory_keys[top_k_indices]
        selected_V = memory_values[top_k_indices]

        # Step 5: Concatenate with local KV and run normal attention
        combined_K = torch.cat([selected_K, local_K], dim=seq_dim)
        combined_V = torch.cat([selected_V, local_V], dim=seq_dim)

        output = scaled_dot_product_attention(Q, combined_K, combined_V)
        return output
```

### Step 2: Encode the rulebook offline

Single forward pass of the rulebook through the model. At each modified layer, capture K and V tensors, chunk-mean pool them (average every 128 tokens into one compressed KV pair), compute a routing key per chunk, and save to disk.

```python
rulebook_text = load_your_rulebook()  # ~20k tokens
tokens = tokenizer(rulebook_text)

with torch.no_grad():
    outputs = model(tokens, output_hidden_states=True)

for layer_idx in modified_layers:
    K, V = outputs.key_values[layer_idx]
    # Chunk-mean pool: every 128 tokens -> 1 compressed token
    K_compressed = K.reshape(-1, 128, head_dim).mean(dim=1)
    V_compressed = V.reshape(-1, 128, head_dim).mean(dim=1)
    K_routing = router_proj(K_compressed)

    save_to_disk(K_compressed, V_compressed, K_routing, layer_idx)
```

A ~20k token rulebook becomes ~156 compressed KV pairs per layer. Tiny footprint.

### Step 3: Fine-tune the router

Only the router projection layers are trained (few million parameters). Base model weights stay frozen. Training data is query→relevant-chunk pairs:

```
Query: "I need to heat an object, what do I do?"
Relevant chunks: [chunk about heating rules, chunk about microwave syntax]

Query: "put apple 1 in/on fridge 1"
Relevant chunks: [chunk about put syntax, chunk about fridge interactions]

Query: "Nothing happens. What went wrong?"
Relevant chunks: [chunk about common errors, chunk about exact ID syntax]
```

~500-1000 pairs needed. Can be generated with an LLM given the rulebook content. Better: use the entropy router logs from Phase 4 — they already tell you which queries got sent to the MSA layer and what the correct routing looked like.

```python
optimizer = Adam(router_parameters_only, lr=1e-4)

for query, target_chunks in training_data:
    routing_scores = model.compute_routing(query, stored_routing_keys)
    loss = cross_entropy(routing_scores, target_chunk_indices)

    # Optional: language modeling loss to preserve generation quality
    lm_output = model.generate_with_memory(query, selected_memory)
    lm_loss = cross_entropy(lm_output, expected_answer)

    total_loss = loss + lm_loss
    total_loss.backward()
    optimizer.step()
```

### Step 4: Inference

```
Agent receives observation
        |
Load pre-computed rulebook KV states (already on GPU, loaded once at startup)
        |
Current prompt = just the observation + recent context (NO system prompt)
        |
Modified attention layers automatically route to relevant rulebook chunks
        |
Model generates action with full rulebook knowledge but only ~200 active tokens
```

The system prompt is gone from the context window entirely.

---

## What to Measure

- **Token savings**: Baseline uses ~20k system prompt + growing history every turn. MSA v2 uses 0 system prompt tokens.
- **Accuracy preservation**: Does the agent still follow rules correctly without reading them as text?
- **Routing quality**: Are the correct rulebook chunks being retrieved? Log which chunks get selected per turn and compare against v1's ChromaDB routing.
- **Latency**: The routing + concatenation adds overhead. Is it less than re-reading the system prompt?

---

## Known Constraints

- **vLLM incompatibility**: Custom attention layers don't work with vLLM's paged-attention kernel. Inference must use the transformers backend or a custom model wrapper. This means MSA v2 benchmarks won't benefit from vLLM's batching/scheduling. Acceptable for a research prototype.
- **Model size ceiling**: Qwen 0.5B may not have enough capacity to follow complex SOPs from latent memory alone. 1.5B is the realistic target; 0.5B is for debugging the attention modification.
- **Training data quality**: The router is only as good as the query→chunk labels. If training pairs are noisy or don't cover edge cases, routing degrades to random chunk selection. Phase 4 entropy router logs are the best source — they capture real agent queries with ground-truth routing decisions.

---

## Recommended Sequencing

Build this **after** Phase 4 (Entropy Router) and Phase 3.5 (Visual Bus + RAG combined). Reasons:

1. The entropy router logs provide free training data for the router projection layers — no synthetic generation needed.
2. The paper's contribution is the composition (modality-to-function mapping + entropy routing), not any single layer's mechanism. Phases 3.5 and 4 deliver the hero results; MSA v2 upgrades one layer from simulation to real.
3. The `MSAStore.query()` interface from v1 is already the drop-in seam. Everything built on top of it (entropy router, multi-agent, dashboard) carries over when the internals swap to real attention.

### Estimated timeline (once prerequisites are done)

- Day 1-2: Modify Qwen 0.5B/1.5B attention layers, implement routing
- Day 3: Encode rulebook, build chunk storage
- Day 4-5: Generate/collect training data, fine-tune router
- Day 6-7: Integrate into agent pipeline, benchmark against v1
