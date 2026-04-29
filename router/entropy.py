"""
Phase 4: Entropy Router.

Decides *which memory layers to consult* on each turn from the Shannon
entropy of a cheap probe pass. The probe is a single-token generation
with no memory injected; its first-token entropy is the agent's
intrinsic uncertainty about the next action given only goal + current
observation.

Routing policy (matches the README's three-band design):

    H < ENTROPY_LOW_THRESHOLD     →  MSA only
        (model is confident — the rulebook + frozen system prompt is
         enough; skip Visual Bus and RAG to save tokens.)

    LOW ≤ H < ENTROPY_MED_THRESHOLD →  MSA + Visual Bus
        (environmental uncertainty — fetch the episodic timeline so the
         model can re-orient itself in the session.)

    H ≥ ENTROPY_MED_THRESHOLD     →  MSA + Visual Bus + RAG
        (epistemic uncertainty about specific facts — also pull exact
         strings from the vector store.)

This module is intentionally pure: it does not touch the LLM, ChromaDB,
or any agent state. The agent owns the probe call and the memory stores;
this just maps a scalar to a decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from configs.settings import (
    ENTROPY_LOW_THRESHOLD,
    ENTROPY_MED_THRESHOLD,
)


# Band labels — kept short because they end up in TurnMetric.memory_source
# and feed straight into dashboard charts.
BAND_LOW = "msa"
BAND_MED = "msa_vbus"
BAND_HIGH = "msa_vbus_rag"


@dataclass
class RouteDecision:
    """The router's output for one turn.

    ``band`` is the coarse label written to ``TurnMetric.memory_source``.
    The boolean flags are what the agent actually keys off when deciding
    which context blocks to assemble.

    ``entropy`` is the probe's first-token entropy (bits). ``-1.0`` means
    the probe was skipped or the backend didn't return logprobs — in that
    case ``band`` falls back to BAND_HIGH so we never silently degrade.

    ``reason`` is a short human-readable trace useful for the dashboard
    and post-hoc analysis.
    """
    entropy: float
    band: str
    use_msa_routed_chunks: bool
    use_visual_bus: bool
    use_rag: bool
    reason: str


def route(
    entropy: float,
    low: float = ENTROPY_LOW_THRESHOLD,
    med: float = ENTROPY_MED_THRESHOLD,
) -> RouteDecision:
    """Map a probe entropy (bits) to a memory-injection decision.

    Negative entropy (no probe / unsupported backend) is treated as
    maximum uncertainty so the agent never silently drops memory layers
    when the signal is missing.
    """
    if entropy < 0:
        return RouteDecision(
            entropy=entropy,
            band=BAND_HIGH,
            use_msa_routed_chunks=True,
            use_visual_bus=True,
            use_rag=True,
            reason="probe unavailable — defaulting to full memory injection",
        )

    if entropy < low:
        return RouteDecision(
            entropy=entropy,
            band=BAND_LOW,
            use_msa_routed_chunks=False,
            use_visual_bus=False,
            use_rag=False,
            reason=f"H={entropy:.3f} < {low} → MSA-only (frozen rulebook)",
        )

    if entropy < med:
        return RouteDecision(
            entropy=entropy,
            band=BAND_MED,
            use_msa_routed_chunks=True,
            use_visual_bus=True,
            use_rag=False,
            reason=f"{low} ≤ H={entropy:.3f} < {med} → MSA + Visual Bus",
        )

    return RouteDecision(
        entropy=entropy,
        band=BAND_HIGH,
        use_msa_routed_chunks=True,
        use_visual_bus=True,
        use_rag=True,
        reason=f"H={entropy:.3f} ≥ {med} → MSA + Visual Bus + RAG",
    )
