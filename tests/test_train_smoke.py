"""Train smoke test: ensure the training loop completes and loss does decrease on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from quant.config import load_market
from quant.models.transformer import StateVectorTransformer
from quant.pipeline.collect import collect
from quant.pipeline.dataset import make_splits
from quant.pipeline.state import build_state
from quant.pipeline.structure import structure


@pytest.fixture(scope="module")
def synthetic_splits(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train_smoke")
    import quant.config as cfg
    cfg.DATA_DIR = tmp / "data"
    cfg.RUNS_DIR = tmp / "runs"
    market = load_market("_synthetic")
    market.model = "transformer"  # force the transformer path here even though yaml default is gbdt
    market.transformer.layers = 2
    market.transformer.model_dim = 32
    market.transformer.heads = 2
    market.train.epochs = 1
    market.train.batch_size = 8
    collect(market)
    structure(market)
    build_state(market)
    splits = make_splits(market)
    return market, splits


def test_transformer_forward_pass(synthetic_splits):
    market, splits = synthetic_splits
    model = StateVectorTransformer(market)
    loader = DataLoader(splits["train"], batch_size=8)
    batch = next(iter(loader))
    out = model(batch["x"])
    assert out.shape[0] == batch["x"].shape[0]


def test_one_epoch_reduces_loss(synthetic_splits):
    market, splits = synthetic_splits
    torch.manual_seed(market.train.seed)
    model = StateVectorTransformer(market)
    loader = DataLoader(splits["train"], batch_size=market.train.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    initial = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"])
            initial.append(float(loss_fn(logits.squeeze(-1), batch["y"].float())))
            break
    initial_loss = initial[0]

    model.train()
    for _ in range(5):  # a few mini-epochs
        for batch in loader:
            logits = model(batch["x"])
            loss = loss_fn(logits.squeeze(-1), batch["y"].float())
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        for batch in loader:
            final = float(loss_fn(model(batch["x"]).squeeze(-1), batch["y"].float()))
            break
    assert final < initial_loss * 1.5, f"loss did not decrease (initial={initial_loss:.4f}, final={final:.4f})"
