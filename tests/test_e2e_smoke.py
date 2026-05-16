"""End-to-end smoke test: collect -> structure -> state -> train -> backtest -> report
on the synthetic market in under 5 minutes locally.

This is the canary that catches pipeline integration bugs before any real market is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant.config import load_market
from quant.pipeline.backtest import backtest
from quant.pipeline.collect import collect
from quant.pipeline.report import report
from quant.pipeline.state import build_state
from quant.pipeline.structure import structure
from quant.pipeline.train import train


@pytest.fixture(scope="module")
def synthetic_pipeline_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    import quant.config as cfg
    cfg.DATA_DIR = tmp / "data"
    cfg.RUNS_DIR = tmp / "runs"

    market = load_market("_synthetic")
    collect(market)
    structure(market)
    build_state(market)
    ckpt = train(market, device="cpu")
    run_dir = ckpt.parent
    metrics = backtest(market, run_dir)
    report(market, run_dir)
    return market, run_dir, metrics


def test_pipeline_runs_end_to_end(synthetic_pipeline_run):
    market, run_dir, metrics = synthetic_pipeline_run
    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "backtest_report.html").exists()
    assert (run_dir / "notes.md").exists()
    assert (run_dir / "equity.png").exists()
    assert (run_dir / "reliability.png").exists()


def test_model_actually_learns_on_synthetic(synthetic_pipeline_run):
    """Pipeline produces non-degenerate predictions: not stuck at 0.5, not constant.

    We do NOT assert the model beats the market on this tiny synthetic set; that question
    is properly answered by the real-market backtests. We DO assert predictions have
    non-trivial variance (so the model is actually using the features) and that there are
    enough held-out events for the backtest to be meaningful.
    """
    import numpy as np
    import pandas as pd
    market, run_dir, metrics = synthetic_pipeline_run
    assert metrics["n_events"] > 10
    trades = pd.read_parquet(run_dir / "trades.parquet")
    assert trades["p_model"].std() > 0.05, "model is producing constant predictions"
    assert trades["p_model"].min() < 0.4 and trades["p_model"].max() > 0.6, \
        "model predictions never cross the decision threshold"


def test_no_nan_in_metrics(synthetic_pipeline_run):
    market, run_dir, metrics = synthetic_pipeline_run
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            assert not np.isnan(v), f"metric {k} is NaN"
