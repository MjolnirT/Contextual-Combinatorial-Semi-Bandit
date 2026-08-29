"""
Online gradient-boosted regression-tree oracle for SquareCB / IGW, using the same backbone as RegCB's ``gb5`` oracle.

Approach
--------
Wrap ``sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=max_depth)`` — the same model class used by ``IncrementalRegressionTreeEnsemble2`` in RegCB — and refit from scratch on RegCB's training schedule:

    training_points = [burn_in + 2^k for k = 0, 1, 2, ...]   (capped at T)

Each ``step()`` appends ``(phi_played, y_played)`` to a FIFO buffer. When ``self.t`` matches one of the training points, the oracle re-fits a fresh ``GradientBoostingRegressor`` on the entire buffer. Between training points predictions come from the most recent fit; before the first fit (i.e. while ``t <= burn_in``) ``predict`` returns zeros, which is uniform under IGW.

This gives SquareCB the exact same regression backbone and refit cadence as RegCB's ``gb5``, modulo the confidence-interval machinery (leaf-Gram pseudo-inverse) which is unused by IGW.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class OnlineGBOracle:
    """``refit`` selects how the model is updated at each training point.

    ``refit=True``  — discard the old ensemble and fit a fresh ``GradientBoostingRegressor(n_estimators=...)`` on **only the latest episode**: the rows collected since the previous training point. The accumulated history is not reused, so the model reflects recent data only. (This is *not* the pre-ablation default, which refit from scratch on the whole buffer — see ``refit=None``.)

    ``refit=False`` — keep the ensemble and extend it via sklearn ``warm_start``: ``n_estimators`` grows by ``n_estimators`` at every training point (100, 200, ..., 1500 over a T=30 000 run) and each ``fit`` runs on the existing buffer (full history by default).

    ``refit=None``  — legacy pre-ablation behavior, kept so the original campaign's traces stay reproducible: fresh model each time, fit on the whole buffer.
    """

    def __init__(self, ld_dim, max_depth=5, n_estimators=100,
                 burn_in=200, T=None, buffer_size=None,
                 normalize_features=False, y_scale=1.0, seed=None,
                 refit=None):
        if y_scale <= 0:
            raise ValueError(f"y_scale must be > 0, got {y_scale}")

        self.ld_dim = int(ld_dim)
        self.max_depth = int(max_depth)
        self.n_estimators = int(n_estimators)
        self.burn_in = int(burn_in)
        self.T = int(T) if T is not None else None
        self.buffer_size = int(buffer_size) if buffer_size is not None else None
        self.normalize_features = bool(normalize_features)
        self.y_scale = float(y_scale)
        self._seed = None if seed is None else int(seed)
        self.refit = refit if refit is None else bool(refit)

        # Rows collected since the previous training point ("latest
        # episode"). Tracked separately from the main buffer so that a
        # bounded buffer_size cannot corrupt the episode slice.
        self._epi_phi = np.empty((0, self.ld_dim), dtype=np.float64)
        self._epi_y = np.empty(0, dtype=np.float64)

        # Schedule of refit rounds. If T is unknown at construction time we
        # extend lazily on demand: the schedule is `burn_in + 2^k` for k >=
        # 0, so we can append more points whenever step_count overtakes the
        # last one.
        self._next_k = 0
        self.training_points: list = []
        self._extend_schedule()

        self.buffer_phi = np.empty((0, self.ld_dim), dtype=np.float64)
        self.buffer_y = np.empty(0, dtype=np.float64)
        self.model: GradientBoostingRegressor | None = None
        self.step_count = 0

    # ------------------------------------------------------------------ #
    def _extend_schedule(self):
        """Append the next doubling point ``burn_in + 2^k`` to the schedule.

        With T known, stop once we exceed T (matching RegCB exactly). Without T, extend lazily — one new point per call."""
        while True:
            t = self.burn_in + (1 << self._next_k)
            if self.T is not None and t > self.T:
                return
            self.training_points.append(t)
            self._next_k += 1
            if self.T is None:
                return

    @staticmethod
    def _l2_normalize_rows(phi):
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        return phi / np.clip(norms, 1e-12, None)

    def _normalize(self, phi):
        f = np.asarray(phi, dtype=np.float64)
        if self.normalize_features:
            f = self._l2_normalize_rows(f)
        return f

    # ------------------------------------------------------------------ #
    def predict(self, features):
        f = self._normalize(features)
        if self.model is None:
            return np.zeros(len(f), dtype=np.float64)
        return self.model.predict(f) * self.y_scale

    def step(self, phi_played, y_played):
        f = self._normalize(phi_played)
        y = np.asarray(y_played, dtype=np.float64) / self.y_scale

        self.buffer_phi = np.concatenate([self.buffer_phi, f], axis=0)
        self.buffer_y = np.concatenate([self.buffer_y, y])
        if self.buffer_size is not None and len(self.buffer_phi) > self.buffer_size:
            self.buffer_phi = self.buffer_phi[-self.buffer_size:]
            self.buffer_y = self.buffer_y[-self.buffer_size:]

        self._epi_phi = np.concatenate([self._epi_phi, f], axis=0)
        self._epi_y = np.concatenate([self._epi_y, y])

        self.step_count += 1

        # Lazy schedule extension when T is unknown.
        if self.T is None:
            while self.training_points and self.step_count > self.training_points[-1]:
                self._extend_schedule()

        if self.step_count not in self._training_set():
            return 0.0

        if self.refit is True:
            # Fresh model on the latest episode only.
            fit_phi, fit_y = self._epi_phi, self._epi_y
            if len(fit_phi) < 2:
                return 0.0          # nothing meaningful to fit on yet
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self._seed,
            )
        elif self.refit is False:
            # Extend the existing ensemble by n_estimators more trees,
            # trained on the buffer (full history unless buffer_size set).
            fit_phi, fit_y = self.buffer_phi, self.buffer_y
            if self.model is None:
                self.model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self._seed,
                    warm_start=True,
                )
            else:
                self.model.n_estimators += self.n_estimators
        else:
            # Legacy: fresh model on the whole buffer.
            fit_phi, fit_y = self.buffer_phi, self.buffer_y
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self._seed,
            )

        self.model.fit(fit_phi, fit_y)
        self._epi_phi = np.empty((0, self.ld_dim), dtype=np.float64)
        self._epi_y = np.empty(0, dtype=np.float64)

        preds = self.model.predict(fit_phi)
        return float(np.mean((fit_y - preds) ** 2))

    # ------------------------------------------------------------------ #
    def _training_set(self):
        # Cache as a set on first use to keep `in` checks O(1). Recomputed
        # whenever the schedule grows (cheap — <= log2(T) entries).
        cached = getattr(self, "_training_set_cache", None)
        if cached is None or len(cached) != len(self.training_points):
            cached = set(self.training_points)
            self._training_set_cache = cached
        return cached
