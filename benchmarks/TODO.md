# SOP-Bench (by Amazon Science, 2026)

This benchmark was literally built to test if LLM agents can follow complex, multi-step industrial rulebooks.

- **What it is:** A benchmark of 2,000+ tasks across 12 industrial domains (e.g., Supply Chain Logistics, Aviation Safety, Content Moderation, Healthcare).
- **The "Haystack":** Every task comes with a massive, authentic human-authored Standard Operating Procedure (SOP) document. For example, a multi-page "Dangerous Goods Classification Manual."
- **The Execution:** The agent is given mock tools and APIs to investigate a specific scenario (e.g., checking a product's safety data sheet) and must output a final classification based strictly on the SOP document.

**How it validates Tri-Mem perfectly:**

- **MSA (Semantic Memory):** The 20-page "Dangerous Goods SOP" is chunked and pre-computed into your MSA layer. Your agent doesn't pay token costs to re-read it every turn, but the entropy router pulls the exact section when the agent is confused about a classification rule.
- **Visual Bus (Working Memory):** As the agent makes 15 different API calls to check flashpoints, package weights, and chemical lists, the text history gets wildly long. The Visual Bus compresses this episodic journey.
- **RAG (Declarative Memory):** Exact chemical formulas, flashpoint temperatures (e.g., "23°C"), and API keys are stored losslessly so they don't blur in the visual compression.
