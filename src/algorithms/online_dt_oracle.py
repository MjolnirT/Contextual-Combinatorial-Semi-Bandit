"""
Online stagewise decision-tree residual oracle for SquareCB / IGW.

Approach
--------
Functional gradient descent in the squared-loss landscape over a sliding replay buffer:

  - Keep a FIFO buffer of the last W (phi, y) pairs seen on played arms.
  - Every ``fit_every`` rounds (after a small warmup), compute residuals
        r = y - F(phi)
    over the buffer, where F is the current ensemble prediction, and fit one new ``sklearn.tree.DecisionTreeRegressor(max_depth)`` to (phi, r). Append it to the ensemble with shrinkage ``learning_rate``.
  - Cap the ensemble at ``max_trees`` (drop oldest first) to bound predict cost as T grows.

Each new tree is one functional-gradient step on the buffered loss — the direct analogue of "one SGD step per round" for the linear oracle, lifted to function space. A buffer is required because a depth-5 tree fit on the L played samples (often L=3) overfits trivially; the buffer gives the weak learner enough rows to find a meaningful split structure.

Note: this is *not* the same as ``OnlineGBOracle`` (sklearn's ``GradientBoostingRegressor`` refit from scratch on RegCB's doubling schedule). This class hand-rolls a stagewise decision-tree residual ensemble and updates online.

This class is not a scikit-learn estimator and does not implement ``fit``/``partial_fit``; it follows the small ``OnlineOracle`` protocol in ``online_oracles.py``.
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor


class OnlineDTOracle:
    def __init__(self, ld_dim, max_depth=5, learning_rate=0.05,
                 buffer_size=2000, fit_every=10, max_trees=200,
                 min_samples_leaf=5, warmup=50,
                 normalize_features=True, y_scale=1.0, seed=None):
        if y_scale <= 0:
            raise ValueError(f"y_scale must be > 0, got {y_scale}")

        self.ld_dim = int(ld_dim)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.fit_every = int(fit_every)
        self.max_trees = int(max_trees)
        self.min_samples_leaf = int(min_samples_leaf)
        self.warmup = int(warmup)
        self.normalize_features = bool(normalize_features)
        self.y_scale = float(y_scale)

        self.rng = np.random.default_rng(seed)
        self.trees: list = []
        self.buffer_phi = np.empty((0, self.ld_dim), dtype=np.float64)
        self.buffer_y = np.empty(0, dtype=np.float64)
        self.step_count = 0

    @staticmethod
    def _l2_normalize_rows(phi):
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        return phi / np.clip(norms, 1e-12, None)

    def _normalize(self, phi):
        f = np.asarray(phi, dtype=np.float64)
        if self.normalize_features:
            f = self._l2_normalize_rows(f)
        return f

    def _ensemble_predict(self, phi):
        # F(phi) in *internal* (y/y_scale) units.
        if not self.trees:
            return np.zeros(len(phi), dtype=np.float64)
        out = np.zeros(len(phi), dtype=np.float64)
        for tree in self.trees:
            out += tree.predict(phi)
        return out * self.learning_rate

    def predict(self, features):
        f = self._normalize(features)
        F = self._ensemble_predict(f)
        # Match the torch oracle's contract: return predictions in the
        # original-y scale so IGW sees scores comparable to true rewards.
        return F * self.y_scale

    def step(self, phi_played, y_played):
        f = self._normalize(phi_played)
        y = np.asarray(y_played, dtype=np.float64) / self.y_scale

        # FIFO append.
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

        # Functional-gradient step: fit one tree to the negative gradient
        # of the squared loss (= residual) over the current buffer.
        F_buf = self._ensemble_predict(self.buffer_phi)
        residual = self.buffer_y - F_buf

        seed = int(self.rng.integers(0, 2**31 - 1))
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
        )
        tree.fit(self.buffer_phi, residual)
        self.trees.append(tree)
        if len(self.trees) > self.max_trees:
            self.trees = self.trees[-self.max_trees:]

        F_new = F_buf + self.learning_rate * tree.predict(self.buffer_phi)
        return float(np.mean((self.buffer_y - F_new) ** 2))
