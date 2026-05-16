"""Stage 6: walk-forward backtest on the held-out window.

Default: train once on train+val (no peek at held-out), then sequentially predict each
held-out event and simulate trading. Commission, slippage, and safety margin are applied
in full. The label-shuffle sanity check (in tests/test_backtest.py) verifies that
shuffling labels collapses Brier improvement to ~0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant.config import MarketConfig
from quant.eval.metrics import (
    BacktestMetrics,
    accuracy,
    brier,
    calibration_bins,
    max_drawdown,
    sharpe,
)
from quant.models.decision import decide
from quant.models.kelly import position_size
from quant.models.transformer import StateVectorTransformer
from quant.pipeline.dataset import make_splits


def backtest(market: MarketConfig, run_dir: Path) -> dict:
    """Run backtest on held-out using the trained checkpoint in `run_dir`."""
    splits = make_splits(market)
    held = splits["held_out"]
    if len(held) == 0:
        raise RuntimeError("held-out split is empty; check held_out_months vs data range")

    p_model = _predict_held_out(market, held, run_dir)
    p_market = held.labels["p_market"].to_numpy()
    y = held.labels["y_realized"].to_numpy().astype(np.float64)
    # Bound p_market to a safe range in case the source had odd values.
    p_market = np.clip(np.nan_to_num(p_market, nan=0.5), 1e-4, 1 - 1e-4)
    p_model = np.clip(p_model, 1e-4, 1 - 1e-4)

    bankroll = market.backtest.starting_bankroll
    equity = [bankroll]
    pnls = []
    trade_count = 0
    hits = 0

    for i in range(len(held)):
        pm = float(p_model[i])
        pkt = float(p_market[i])
        outcome = float(y[i])
        decision = decide(
            pm,
            pkt,
            commission_bps=market.backtest.commission_bps,
            base_slippage_bps=market.backtest.base_slippage_bps,
            safety_margin_bps=market.backtest.safety_margin_bps,
        )
        if decision.side == "SKIP":
            equity.append(bankroll)
            continue

        decimal_odds = 1.0 / pkt
        if decision.side == "BACK":
            stake = position_size(
                pm, decimal_odds, bankroll,
                kelly_fraction_ratio=market.backtest.kelly_fraction,
                kelly_cap=market.backtest.kelly_cap,
            )
            if stake <= 0:
                equity.append(bankroll)
                continue
            # P&L: win = stake * (decimal_odds - 1) - commission; loss = -stake
            commission = stake * (market.backtest.commission_bps / 10_000.0)
            slip = stake * (market.backtest.base_slippage_bps / 10_000.0)
            if outcome >= 0.5:
                pnl = stake * (decimal_odds - 1) - commission - slip
                hits += 1
            else:
                pnl = -stake - commission - slip
        else:  # LAY
            decimal_lay_odds = 1.0 / (1.0 - pkt)
            stake = position_size(
                1 - pm, decimal_lay_odds, bankroll,
                kelly_fraction_ratio=market.backtest.kelly_fraction,
                kelly_cap=market.backtest.kelly_cap,
            )
            if stake <= 0:
                equity.append(bankroll)
                continue
            commission = stake * (market.backtest.commission_bps / 10_000.0)
            slip = stake * (market.backtest.base_slippage_bps / 10_000.0)
            if outcome < 0.5:
                pnl = stake * (decimal_lay_odds - 1) - commission - slip
                hits += 1
            else:
                pnl = -stake - commission - slip

        bankroll += pnl
        pnls.append(pnl)
        equity.append(bankroll)
        trade_count += 1

    metrics = BacktestMetrics(
        brier_model=brier(p_model, y),
        brier_market=brier(p_market, y),
        brier_improvement=brier(p_market, y) - brier(p_model, y),
        accuracy_model=accuracy(p_model, y),
        accuracy_market=accuracy(p_market, y),
        n_events=len(held),
        n_trades=trade_count,
        hit_rate=(hits / trade_count) if trade_count > 0 else 0.0,
        gross_return=(bankroll / market.backtest.starting_bankroll) - 1,
        net_return=(bankroll / market.backtest.starting_bankroll) - 1,
        sharpe=sharpe(np.asarray(pnls)),
        max_drawdown=max_drawdown(np.asarray(equity)),
        calibration_bins=calibration_bins(p_model, y),
    )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, default=str))

    # Persist trades for later inspection.
    trades = pd.DataFrame({
        "p_model": p_model,
        "p_market": p_market,
        "y_realized": y,
        "equity_after": equity[1:],
    })
    trades.to_parquet(run_dir / "trades.parquet", index=False)
    return metrics.to_dict()


def _predict_held_out(market: MarketConfig, held_ds, run_dir: Path) -> np.ndarray:
    sequences = held_ds.sequences
    if market.model == "transformer":
        ckpt_path = run_dir / "ckpt.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = StateVectorTransformer(market)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(sequences).float()
            logits = model(x)
            if market.label.kind == "binary":
                return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # For bucketed labels we convert to a single "above-median bucket" probability
            # so the backtest decision rule has a scalar to compare against p_market.
            half = probs.shape[1] // 2
            return probs[:, half:].sum(axis=1)
    # GBDT
    import pickle
    with (run_dir / "model.lgb.pkl").open("rb") as f:
        wrapper = pickle.load(f)
    p = wrapper.predict_proba(sequences)
    if p.ndim == 2:
        half = p.shape[1] // 2
        return p[:, half:].sum(axis=1)
    return p
