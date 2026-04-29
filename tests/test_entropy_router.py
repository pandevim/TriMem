"""Unit tests for `router.entropy.route()`.

Pinning the routing logic so threshold-edge bugs surface in CI rather
than during a benchmark run. Run with:

    pytest tests/test_entropy_router.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

# Allow `pytest tests/...` to find the project modules without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import ENTROPY_LOW_THRESHOLD as LOW
from configs.settings import ENTROPY_MED_THRESHOLD as MED
from router.entropy import BAND_HIGH, BAND_LOW, BAND_MED, route


def test_thresholds_are_ordered():
    assert 0 <= LOW < MED


def test_negative_entropy_defaults_to_high_band():
    # Probe unavailable → never silently degrade memory injection.
    d = route(-1.0)
    assert d.band == BAND_HIGH
    assert d.use_msa_routed_chunks
    assert d.use_visual_bus
    assert d.use_rag


def test_well_below_low_picks_low_band():
    # Per paper §3.4: LOW band is "MSA only" — MSA chunks stay on as a
    # cheap rulebook safety net, Visual Bus + RAG are skipped to save
    # tokens on confident turns.
    d = route(LOW / 2)
    assert d.band == BAND_LOW
    assert d.use_msa_routed_chunks is True
    assert not d.use_visual_bus
    assert not d.use_rag


def test_at_low_threshold_falls_into_med():
    # Boundary: `entropy < low` is strict, so H == LOW lands in MED.
    d = route(LOW)
    assert d.band == BAND_MED
    assert d.use_visual_bus
    assert not d.use_rag


def test_between_thresholds_picks_med_band():
    d = route((LOW + MED) / 2)
    assert d.band == BAND_MED
    assert d.use_msa_routed_chunks
    assert d.use_visual_bus
    assert not d.use_rag


def test_at_med_threshold_falls_into_high():
    # Boundary: `entropy < med` is strict, so H == MED lands in HIGH.
    d = route(MED)
    assert d.band == BAND_HIGH
    assert d.use_rag


def test_well_above_med_picks_high_band():
    d = route(MED + 1.0)
    assert d.band == BAND_HIGH
    assert d.use_msa_routed_chunks
    assert d.use_visual_bus
    assert d.use_rag


@pytest.mark.parametrize(
    "entropy,expected_band",
    [
        (-1.0, BAND_HIGH),
        (-0.001, BAND_HIGH),
        (0.0, BAND_LOW),
        (LOW - 1e-6, BAND_LOW),
        (LOW, BAND_MED),
        (MED - 1e-6, BAND_MED),
        (MED, BAND_HIGH),
        (10.0, BAND_HIGH),
    ],
)
def test_band_table(entropy, expected_band):
    assert route(entropy).band == expected_band


@pytest.mark.parametrize(
    "entropy,expected_band",
    [
        # Spot-checks against the trimem 18-task run (logs/trimem_1777427808.json).
        (1.1451, BAND_HIGH),  # task 1, turn 0 — cold start
        (0.7447, BAND_HIGH),  # task 0, turn 0
        (0.6639, BAND_MED),   # task 0, turn 5 — just under MED
        (0.3402, BAND_MED),   # task 1, turn 2 — minimum observed in that run
        (0.7910, BAND_HIGH),  # task 5, final-upload entropy spike
    ],
)
def test_regression_observed_entropies(entropy, expected_band):
    assert route(entropy).band == expected_band


def test_route_is_pure():
    # Calling route() must not mutate global state — call twice, same answer.
    a = route(0.5)
    b = route(0.5)
    assert a.band == b.band
    assert a.entropy == b.entropy
    assert a.use_visual_bus == b.use_visual_bus


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
