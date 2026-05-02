"""
Online XGBoost regression-tree oracle for SquareCB / IGW.

Approach
--------
Same replay-buffer functional-gradient recipe as ``OnlineGBOracle``, but
the tree builder is XGBoost with native L1/L2 leaf regularization, the
hist algorithm, and accumulated booster state across calls:

  - FIFO replay buffer of recent (phi, y) pairs (size W).
  - Every ``fit_every`` rounds (after ``warmup`` samples), rebuild the
    DMatrix from the buffer and call
        xgb.train(params, dmat,
                  num_boost_round=num_boost_round,
                  xgb_model=self.booster)
    which appends ``num_boost_round`` new trees to the running booster.
    XGBoost handles the squared-error gradient internally; ``learning_rate``
    is the per-tree shrinkage applied inside XGBoost (do not double-shrink).
  - Predict via ``booster.predict(DMatrix(features))``, scaled back to
    original-y units.

Why this differs from OnlineGBOracle:
  - sklearn DecisionTreeRegressor has no leaf regularization; XGBoost has
    ``reg_alpha`` / ``reg_lambda`` (L1/L2) and ``min_child_weight``, which
    matter when each new tree is fit on noisy single-round residuals.
  - XGBoost predict is C++ and stays cheap as the ensemble grows past
    sklearn's Python-loop break-even point (~hundreds of trees).
"""

import numpy as np
import xgboost as xgb


class OnlineXGBOracle:
    def __init__(self, ld_dim, max_depth=5, learning_rate=0.05,
                 buffer_size=2000, fit_every=10, num_boost_round=1,
                 reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0,
                 subsample=1.0, colsample_bytree=1.0,
                 tree_method='hist', warmup=50,
                 normalize_features=True, y_scale=1.0, seed=None):
        if y_scale <= 0:
            raise ValueError(f"y_scale must be > 0, got {y_scale}")

        self.ld_dim = int(ld_dim)
        self.num_boost_round = int(num_boost_round)
        self.buffer_size = int(buffer_size)
        self.fit_every = int(fit_every)
        self.warmup = int(warmup)
        self.normalize_features = bool(normalize_features)
        self.y_scale = float(y_scale)

        self._base_params = {
            'max_depth': int(max_depth),
            'learning_rate': float(learning_rate),
            'objective': 'reg:squarederror',
            'reg_lambda': float(reg_lambda),
            'reg_alpha': float(reg_alpha),
            'min_child_weight': float(min_child_weight),
            'subsample': float(subsample),
            'colsample_bytree': float(colsample_bytree),
            'tree_method': tree_method,
            'verbosity': 0,
        }

        self.rng = np.random.default_rng(seed)
        self.booster: xgb.Booster | None = None
        self.buffer_phi = np.empty((0, self.ld_dim), dtype=np.float32)
        self.buffer_y = np.empty(0, dtype=np.float32)
        self.step_count = 0

    @staticmethod
    def _l2_normalize_rows(phi):
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        return phi / np.clip(norms, 1e-12, None)

    def _normalize(self, phi):
        f = np.asarray(phi, dtype=np.float32)
        if self.normalize_features:
            f = self._l2_normalize_rows(f)
        return f

    def predict(self, features):
        f = self._normalize(features)
        if self.booster is None:
            return np.zeros(len(f), dtype=np.float64)
        F = self.booster.predict(xgb.DMatrix(f)).astype(np.float64)
        return F * self.y_scale

    def step(self, phi_played, y_played):
        f = self._normalize(phi_played)
        y = np.asarray(y_played, dtype=np.float32) / self.y_scale

        self.buffer_phi = np.concatenate([self.buffer_phi, f], axis=0)
        self.buffer_y = np.concatenate([self.buffer_y, y])
        if len(self.buffer_phi) > self.buffer_size:
            self.buffer_phi = self.buffer_phi[-self.buffer_size:]
            self.buffer_y = self.buffer_y[-self.buffer_size:]

        self.step_count += 1
        if len(self.buffer_phi) < self.warmup:
            return 0.0
        if self.step_count % self.fit_every != 0:
            return 0.0

        params = dict(self._base_params)
        params['seed'] = int(self.rng.integers(0, 2**31 - 1))

        dmat = xgb.DMatrix(self.buffer_phi, label=self.buffer_y)
        self.booster = xgb.train(
            params, dmat,
            num_boost_round=self.num_boost_round,
            xgb_model=self.booster,
        )

        F_new = self.booster.predict(dmat)
        return float(np.mean((self.buffer_y - F_new) ** 2))
