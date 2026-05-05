"""
RegCB (Foster, Agarwal, Dudík, Luo, Schapire — ICML 2018):
"Practical Contextual Bandits with Regression Oracles".

Restricted to the contextual bandit setting (L=1). Two modes:

  * ``optimistic``  : play  argmax_a HIGH_t(x, a)
  * ``elimination`` : sample uniformly from
                      { a : HIGH_t(x, a) >= max_{a'} LOW_t(x, a') }

Two oracle-refit regimes, selected by ``regcb_refit_per_step``:

  * False (DEFAULT, "lazy" / FALCON-style):
        Refit the regression oracle ONLY at exponential epochs (powers
        of two after warmup); cache the model and its baseline mean-
        squared loss. Per-round HIGH / LOW are then closed-form bands
        around the cached prediction, with no per-action refits:

            HIGH(z) = f_hat(z) + sqrt(beta * baseline_loss)
            LOW(z)  = f_hat(z) - sqrt(beta * baseline_loss)

        This collapses per-round cost from O(K · binsearch_steps ·
        oracle_fit) to O(K · oracle_predict). It is the only practical
        regime for the gb5 oracle.

  * True ("eager" / paper-faithful generic AMO):
        Run the binary-search HIGH / LOW procedure on every round.
        For a candidate feature row z and reward target v, refit the
        oracle on history + one synthetic weighted point (z, v); v is
        feasible iff the resulting average squared loss on the original
        history stays within ``beta`` of the epoch baseline. HIGH(z)
        is the largest feasible v, LOW(z) the smallest. Slow (per-
        round cost grows with history); kept for ablation only.

Two function classes are supported:

  * ``lin``  : ``sklearn.linear_model.Ridge(alpha=ridge_alpha,
               fit_intercept=False)`` -- regularized least squares.
  * ``gb5``  : ``sklearn.ensemble.GradientBoostingRegressor(
               n_estimators=regcb_gb_n_estimators, max_depth=5)``.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .base import Semibandit


class RegCB(Semibandit):
    """RegCB binary-search variant restricted to L=1."""

    def __init__(self, B, learning_alg=None, classification=False):
        if classification:
            raise NotImplementedError("RegCB supports per-arm regression only.")
        if int(getattr(B, "L", 1)) != 1:
            raise ValueError(
                "RegCB requires L=1 (contextual bandit). Got B.L=%d" % int(B.L)
            )
        self.B = B
        self.learning_alg = learning_alg

    def init(self, T, params=None):
        params = params or {}

        self.mode = params.get("regcb_mode", "optimistic")
        if self.mode not in ("optimistic", "elimination"):
            raise ValueError(f"Unknown regcb_mode: {self.mode!r}")

        # ``beta`` is the confidence slack added to the baseline loss; the
        # CLI surfaces it as ``--param``.
        self.beta = float(params.get("beta", params.get("param", 0.1)))

        self.warmup = int(params.get("regcb_warmup", 50))
        self.binsearch_steps = int(params.get("regcb_binsearch_steps", 20))
        self.synth_weight = float(params.get("regcb_synth_weight", 1.0))
        self.reward_min = float(params.get("regcb_reward_min", 0.0))
        self.reward_max = float(params.get("regcb_reward_max", 4.0))
        self.ridge_alpha = float(params.get("regcb_ridge_alpha", 1.0))
        # gb5 ensemble size; 100 is the paper-style default but smaller
        # ensembles are better-calibrated at small history sizes and ~N×
        # cheaper per binary-search step.
        self.gb_n_estimators = int(params.get("regcb_gb_n_estimators", 100))
        # Default: refit only at exponential epochs (FALCON-style). Set
        # True to use the original generic AMO binary search per round.
        self.refit_per_step = bool(params.get("regcb_refit_per_step", False))

        seed = params.get("seed", None)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # ``learning_alg`` is the function-class selector. Only 'lin' and
        # 'gb5' are supported in v1; the harness sets it via --learning_alg.
        name = params.get("learning_alg_name", self.learning_alg)
        if isinstance(name, str):
            self._oracle_name = name
        elif callable(self.learning_alg):
            # The harness sometimes passes a factory lambda for legacy algos;
            # extract the name from a CLI marker if present.
            self._oracle_name = params.get("learning_alg_name", "lin")
        else:
            self._oracle_name = "lin"
        if self._oracle_name not in ("lin", "gb5"):
            raise ValueError(
                f"RegCB v1 supports learning_alg in {{'lin','gb5'}}, got {self._oracle_name!r}"
            )

        # Played-arm history: each row is the per-arm feature of the arm
        # actually played; each y is the observed reward.
        self.history_phi: list = []
        self.history_y: list = []

        # Epoch state: baseline ERM model + average squared loss, refit at
        # geometrically-spaced epochs after warmup.
        self.epoch_baseline_loss = float("inf")
        self.epoch_model = None
        self.next_epoch_n = max(self.warmup, 4)

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None

    # ------------------------------------------------------------------ #
    # Oracle
    # ------------------------------------------------------------------ #
    def _make_oracle(self):
        if self._oracle_name == "lin":
            return Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        # gb5
        return GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators, max_depth=5,
            random_state=None if self.seed is None else int(self.seed),
        )

    def _fit(self, X, y, w=None):
        m = self._make_oracle()
        if w is not None:
            m.fit(X, y, sample_weight=w)
        else:
            m.fit(X, y)
        return m

    # ------------------------------------------------------------------ #
    # Epoch baseline
    # ------------------------------------------------------------------ #
    def _maybe_refresh_epoch(self):
        n = len(self.history_phi)
        if n < self.next_epoch_n:
            return
        X = np.asarray(self.history_phi, dtype=np.float64)
        y = np.asarray(self.history_y, dtype=np.float64)
        m = self._fit(X, y)
        yp = np.asarray(m.predict(X), dtype=np.float64)
        self.epoch_baseline_loss = float(np.mean((yp - y) ** 2))
        self.epoch_model = m
        # Geometric epoch schedule: next epoch when the history doubles.
        self.next_epoch_n = max(self.next_epoch_n * 2, n + 1)

    # ------------------------------------------------------------------ #
    # HIGH / LOW via binary search
    # ------------------------------------------------------------------ #
    def _feasible(self, z, v, X, y, target_loss):
        """True iff refitting on (X, y) plus synthetic point (z, v) gives
        average squared loss on (X, y) <= ``target_loss``."""
        X_aug = np.vstack([X, z[None, :]])
        y_aug = np.append(y, v)
        n = len(y)
        w = np.empty(n + 1, dtype=np.float64)
        w[:n] = 1.0
        w[n] = self.synth_weight
        m = self._fit(X_aug, y_aug, w=w)
        yp = np.asarray(m.predict(X), dtype=np.float64)
        return float(np.mean((yp - y) ** 2)) <= target_loss

    def _radius(self):
        """FALCON-style residual-based confidence radius: sqrt(beta * baseline_loss).

        Tightens as the cached model's baseline MSE shrinks; scales with sqrt(beta).
        """
        return float(np.sqrt(self.beta * max(self.epoch_baseline_loss, 1e-12)))

    def _pred_lazy(self, z):
        return float(self.epoch_model.predict(z[None, :])[0])

    def _high(self, z, X=None, y=None, target_loss=None):
        """Largest plausible reward at z.

        Lazy mode (default): cached_pred(z) + radius, clipped to [reward_min,
        reward_max]. No refit.
        Eager mode: paper-faithful binary search; refits per step.
        """
        if not self.refit_per_step:
            return float(min(self.reward_max,
                             max(self.reward_min, self._pred_lazy(z) + self._radius())))
        # Eager / paper-faithful binary search.
        target_loss = self.epoch_baseline_loss + self.beta if target_loss is None else target_loss
        lo, hi = self.reward_min, self.reward_max
        if self._feasible(z, hi, X, y, target_loss):
            return hi
        if not self._feasible(z, lo, X, y, target_loss):
            return self._pred_lazy(z)
        for _ in range(self.binsearch_steps):
            mid = 0.5 * (lo + hi)
            if self._feasible(z, mid, X, y, target_loss):
                lo = mid
            else:
                hi = mid
        return lo

    def _low(self, z, X=None, y=None, target_loss=None):
        """Smallest plausible reward at z."""
        if not self.refit_per_step:
            return float(max(self.reward_min,
                             min(self.reward_max, self._pred_lazy(z) - self._radius())))
        target_loss = self.epoch_baseline_loss + self.beta if target_loss is None else target_loss
        lo, hi = self.reward_min, self.reward_max
        if self._feasible(z, lo, X, y, target_loss):
            return lo
        if not self._feasible(z, hi, X, y, target_loss):
            return self._pred_lazy(z)
        for _ in range(self.binsearch_steps):
            mid = 0.5 * (lo + hi)
            if self._feasible(z, mid, X, y, target_loss):
                hi = mid
            else:
                lo = mid
        return hi

    # ------------------------------------------------------------------ #
    # Semibandit interface
    # ------------------------------------------------------------------ #
    def get_action(self, x):
        K = int(x.get_K())
        features = np.asarray(x.get_ld_features(), dtype=np.float64)

        # During warmup -- before the first epoch -- play uniformly at random.
        if self.epoch_model is None:
            a = int(self.rng.integers(0, K))
            self.action = np.array([a], dtype=np.int64)
            return self.action

        # Eager (paper-faithful binary search) needs the full history per
        # call to ``_feasible``; lazy mode reads only the cached epoch
        # model so we skip the np.asarray rebuild on every round.
        if self.refit_per_step:
            X = np.asarray(self.history_phi, dtype=np.float64)
            y = np.asarray(self.history_y, dtype=np.float64)
            target_loss = self.epoch_baseline_loss + self.beta
        else:
            X = y = target_loss = None

        highs = np.empty(K, dtype=np.float64)
        for a in range(K):
            highs[a] = self._high(features[a], X, y, target_loss)

        if self.mode == "optimistic":
            a = int(np.argmax(highs))
        else:
            lows = np.empty(K, dtype=np.float64)
            for a_idx in range(K):
                lows[a_idx] = self._low(features[a_idx], X, y, target_loss)
            best_low = float(np.max(lows))
            kept = np.where(highs >= best_low - 1e-12)[0]
            if kept.size == 0:
                kept = np.array([int(np.argmax(highs))], dtype=np.int64)
            a = int(self.rng.choice(kept))

        self.action = np.array([a], dtype=np.int64)
        return self.action

    def update(self, x, A, y_vec, r):
        del r  # RegCB uses the per-arm reward of the played arm.
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        a = int(np.asarray(A, dtype=np.int64).ravel()[0])
        y = float(np.asarray(y_vec, dtype=np.float64).ravel()[0])
        self.history_phi.append(features[a].copy())
        self.history_y.append(y)
        self._maybe_refresh_epoch()
        self.action = None
        self.t += 1
