"""Backtest sanity checks.

1. Backtest runs end-to-end on synthetic data with no NaN.
2. Label-shuffle sanity check: shuffling labels collapses Brier improvement to near zero.
   If the pipeline has a data leak, the shuffle test will still show a positive Brier
   improvement and we want to know immediately.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.config import load_market
from quant.pipeline.backtest import backtest
from quant.pipeline.collect import collect
from quant.pipeline.dataset import make_splits
from quant.pipeline.state import build_state
from quant.pipeline.structure import structure
from quant.pipeline.train import train
from quant.models.gbdt import GBDTWrapper


@pytest.fixture(scope="module")
def trained_synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("backtest")
    import quant.config as cfg
    cfg.DATA_DIR = tmp / "data"
    cfg.RUNS_DIR = tmp / "runs"
    market = load_market("_synthetic")
    collect(market)
    structure(market)
    build_state(market)
    run_dir = train(market, device="cpu")
    return market, run_dir.parent


def test_backtest_runs_and_has_metrics(trained_synthetic):
    market, run_dir = trained_synthetic
    metrics = backtest(market, run_dir)
    for k in ("brier_model", "brier_market", "brier_improvement", "n_events", "n_trades", "sharpe", "max_drawdown"):
        assert k in metrics, f"missing metric: {k}"
    assert not np.isnan(metrics["brier_model"])


def test_label_shuffle_collapses_brier_improvement(tmp_path_factory):
    """Retrain with shuffled labels; Brier improvement vs market should collapse near 0."""
    tmp = tmp_path_factory.mktemp("backtest_shuffle")
    import quant.config as cfg
    cfg.DATA_DIR = tmp / "data"
    cfg.RUNS_DIR = tmp / "runs"
    market = load_market("_synthetic")
    collect(market)
    structure(market)
    build_state(market)

    # Shuffle labels in the train+val partition only (DO NOT shuffle held-out).
    splits = make_splits(market)
    rng = np.random.default_rng(123)
    shuffled_targets = splits["train"].targets.copy()
    rng.shuffle(shuffled_targets)
    splits["train"].targets = shuffled_targets

    # Fit a GBDT on shuffled labels.
    wrapper = GBDTWrapper(market)
    wrapper.fit(splits["train"].sequences, splits["train"].targets, splits["val"].sequences, splits["val"].targets)

    # Predict held-out and check Brier vs market.
    p_model = wrapper.predict_proba(splits["held_out"].sequences)
    if p_model.ndim == 2:
        p_model = p_model[:, 1]
    p_market = splits["held_out"].labels["p_market"].to_numpy()
    y = splits["held_out"].labels["y_realized"].to_numpy().astype(float)
    brier_model = float(np.mean((p_model - y) ** 2))
    brier_market = float(np.mean((p_market - y) ** 2))
    improvement = brier_market - brier_model
    # With shuffled labels the model should not beat the market by more than a small
    # statistical fluctuation.
    assert improvement < 0.02, f"shuffle collapse failed: brier improvement = {improvement:.4f}"
