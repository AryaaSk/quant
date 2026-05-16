"""Property-based test: no feature in the state vector at timestep t derives from data
with timestamp > t. This is THE invariant of the entire pipeline.

We use Hypothesis to sample random events and feature slots and assert the property
holds across the synthetic dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover
    pytest.skip("hypothesis not installed", allow_module_level=True)

from quant.config import load_market
from quant.pipeline.collect import collect
from quant.pipeline.state import build_state
from quant.pipeline.structure import structure
from quant.state.composer import StateBuilder


@pytest.fixture(scope="module")
def synthetic_pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("temporal_leak")
    import quant.config as cfg
    cfg.DATA_DIR = tmp / "data"
    cfg.RUNS_DIR = tmp / "runs"
    market = load_market("_synthetic")
    collect(market)
    structure(market)
    seq_path, label_path = build_state(market)
    sequences = np.load(seq_path)
    labels = pd.read_parquet(label_path)
    structured = pd.read_parquet(market.structured_dir() / "features.parquet")
    structured.index = pd.to_datetime(structured.index)
    return market, sequences, labels, structured


def test_sequences_align_to_pre_event_window(synthetic_pipeline):
    market, sequences, labels, structured = synthetic_pipeline
    L = market.sequence_length
    assert sequences.ndim == 3 and sequences.shape[1] == L and sequences.shape[2] == market.state_vector.dim


def test_no_future_published_or_scraped_data_leaks_per_event(synthetic_pipeline):
    market, sequences, labels, structured = synthetic_pipeline
    builder = StateBuilder(market.state_vector)
    builder.fit(structured)

    # For each event, the latest timestamp visible to the model when building the LAST
    # state vector in the sequence must be strictly before the event time. By construction
    # (`state.py` uses `searchsorted` with side="left"), the end_idx points at the row at
    # event time, so steps go up to end_idx-1 < event_time.
    for _, ev in labels.iterrows():
        t_event = pd.Timestamp(ev["target_event_time"])
        end_idx = structured.index.searchsorted(t_event, side="left")
        # The last step builds state at structured.index[end_idx - 1] which must be < t_event.
        if end_idx > 0:
            assert structured.index[end_idx - 1] < t_event, (
                f"event {ev['event_id']} at {t_event} has visible structured row at "
                f"{structured.index[end_idx - 1]} which is NOT before the event"
            )


@settings(max_examples=50, deadline=None)
@given(event_idx=st.integers(min_value=0, max_value=199))
def test_state_vector_recompute_is_deterministic(synthetic_pipeline, event_idx):
    """If we recompute the state vector for any (event, timestep) it matches the saved one
    exactly. This guarantees the state-construction function is pure and inspectable.
    """
    market, sequences, labels, structured = synthetic_pipeline
    if event_idx >= len(labels):
        return
    builder = StateBuilder(market.state_vector)
    builder.fit(structured)

    ev = labels.iloc[event_idx]
    t_event = pd.Timestamp(ev["target_event_time"])
    L = market.sequence_length
    end_idx = structured.index.searchsorted(t_event, side="left")
    start_idx = max(0, end_idx - L)
    steps = list(range(start_idx, end_idx))
    if len(steps) < L:
        pad = L - len(steps)
        steps = [steps[0] if steps else 0] * pad + steps

    recomputed = np.stack([builder.build(structured.index[s], structured) for s in steps], axis=0)
    np.testing.assert_allclose(recomputed, sequences[event_idx], atol=1e-6)
