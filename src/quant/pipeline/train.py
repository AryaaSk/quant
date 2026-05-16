"""Stage 5: train the model.

Defaults to a transformer; falls back to LightGBM via market.model = "gbdt" (which is set
automatically by the screening + dataset-size heuristic upstream).

Writes runs/<slug>/<timestamp>/ckpt.pt (transformer) or model.lgb (GBDT) and a train.log.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quant.config import MarketConfig
from quant.models.gbdt import GBDTWrapper
from quant.models.transformer import StateVectorTransformer
from quant.pipeline.dataset import make_splits


def _now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def train(market: MarketConfig, *, out_dir: Path | None = None, device: str = "auto") -> Path:
    if out_dir is None:
        out_dir = market.runs_dir() / _now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(market)

    if market.model == "transformer":
        ckpt = _train_transformer(market, splits, out_dir, device=device)
    elif market.model == "gbdt":
        ckpt = _train_gbdt(market, splits, out_dir)
    else:
        raise ValueError(f"unknown model: {market.model}")

    (out_dir / "train.meta.json").write_text(json.dumps({
        "model": market.model,
        "slug": market.slug,
        "device": device,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_held_out": len(splits["held_out"]),
        "state_dim": market.state_vector.dim,
        "sequence_length": market.sequence_length,
    }, indent=2))
    return ckpt


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _train_transformer(market: MarketConfig, splits, out_dir: Path, device: str) -> Path:
    dev = _resolve_device(device)
    model = StateVectorTransformer(market).to(dev)
    train_loader = DataLoader(splits["train"], batch_size=market.train.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(splits["val"], batch_size=market.train.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=market.train.learning_rate, weight_decay=market.train.weight_decay)
    if market.label.kind == "binary":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    log_path = out_dir / "train.log"
    log_lines: list[str] = []
    best_val = math.inf
    ckpt_path = out_dir / "ckpt.pt"

    for epoch in range(market.train.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(dev)
            y = batch["y"].to(dev)
            logits = model(x)
            if market.label.kind == "binary":
                loss = loss_fn(logits.squeeze(-1), y.float())
            else:
                loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), market.train.grad_clip)
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(dev)
                y = batch["y"].to(dev)
                logits = model(x)
                if market.label.kind == "binary":
                    loss = loss_fn(logits.squeeze(-1), y.float())
                else:
                    loss = loss_fn(logits, y)
                val_losses.append(float(loss.item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        vl = float(np.mean(val_losses)) if val_losses else float("nan")
        log_lines.append(f"epoch={epoch:03d} train_loss={tr:.4f} val_loss={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save({"model_state": model.state_dict(), "market_slug": market.slug}, ckpt_path)

    log_path.write_text("\n".join(log_lines) + "\n")
    return ckpt_path


def _train_gbdt(market: MarketConfig, splits, out_dir: Path) -> Path:
    import pickle
    wrapper = GBDTWrapper(market)
    train_ds = splits["train"]
    val_ds = splits["val"]

    train_seq = train_ds.sequences
    train_tgt = train_ds.targets
    val_seq = val_ds.sequences
    val_tgt = val_ds.targets

    wrapper.fit(train_seq, train_tgt, val_seq, val_tgt)
    ckpt_path = out_dir / "model.lgb.pkl"
    with ckpt_path.open("wb") as f:
        pickle.dump(wrapper, f)
    return ckpt_path
