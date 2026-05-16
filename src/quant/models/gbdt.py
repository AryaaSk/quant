"""GBDT fallback for low-event-count markets.

Uses scikit-learn's HistGradientBoosting (bundled, no native deps, robust on macOS arm64).
Flattens the state-vector sequence (concat all timesteps into one row) and trains a
binary or multiclass classifier. Same downstream interface: `predict_proba(sequences)`.

Note: we tried LightGBM 4.x first; it segfaults on macOS arm64 + numpy 2.4+ in
`_init_from_np2d`. sklearn HistGradientBoosting is the closest like-for-like substitute
with no native compilation surface to fight.
"""
from __future__ import annotations

import numpy as np

from quant.config import MarketConfig


class GBDTWrapper:
    def __init__(self, market: MarketConfig):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as e:  # pragma: no cover
            raise ImportError("scikit-learn is required for the GBDT fallback") from e
        self._cls = HistGradientBoostingClassifier
        self.market = market
        self.model = None

    def fit(self, sequences: np.ndarray, targets: np.ndarray, val_seq: np.ndarray | None = None, val_tgt: np.ndarray | None = None):
        X = sequences.reshape(sequences.shape[0], -1).astype(np.float32, copy=False)
        y = targets.astype(np.int64 if self.market.label.kind != "binary" else np.int64)
        if self.market.label.kind == "binary":
            y = (np.asarray(targets) > 0.5).astype(np.int64)
        self.model = self._cls(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=5,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.15 if val_seq is None else None,
            n_iter_no_change=30,
            random_state=self.market.train.seed,
        )
        if val_seq is not None and val_tgt is not None and len(val_seq) > 0:
            # sklearn's HGB does its own internal validation split; concatenate and let it work.
            Xv = val_seq.reshape(val_seq.shape[0], -1).astype(np.float32, copy=False)
            yv = (np.asarray(val_tgt) > 0.5).astype(np.int64) if self.market.label.kind == "binary" else val_tgt.astype(np.int64)
            X_all = np.concatenate([X, Xv], axis=0)
            y_all = np.concatenate([y, yv], axis=0)
            self.model.fit(X_all, y_all)
        else:
            self.model.fit(X, y)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        X = sequences.reshape(sequences.shape[0], -1).astype(np.float32, copy=False)
        p = self.model.predict_proba(X)
        # Sklearn always returns (N, n_classes); convert binary to (N,) on positive class.
        if self.market.label.kind == "binary":
            return p[:, 1]
        return p
