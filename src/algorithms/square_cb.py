import numpy as np

from Policy import RegressionPolicy

from typing import Optional

from .base import Semibandit
from .igw_sampler import (default_coverage, depround_slate,
                          igw_bandit_sample_slate, igw_marginals, sample_slate)
from .online_oracles import OnlineOracle, make_oracle


# Param keys consumed by SquareCB itself (not forwarded to the oracle).
# ``model_type`` is the oracle selector and is read separately.
# Legacy keys (``weight``, ``mu``, ``eps``, ``delta``, ``train_all``) come
# from non-sqcb algorithms reusing this same params dict path; ignore them.
_SQUARECB_KEYS = frozenset({
    'gamma_multiplier', 'gamma_schedule', 'coverage', 'solver',
    'sampler', 'logger', 'opt_against', 'log_every', 'model_type',
    'weight', 'mu', 'eps', 'delta', 'train_all',
})


class SquareCB(Semibandit):
    """
    SquareCB / IGW for the L-of-K semi-bandit setting with online
    per-arm regression updates.

    Each round:
      1. Predict per-item rewards f_hat(phi_k) for all K arms via the
         online regressor.
      2. IGW turns scores into marginal inclusion probabilities p_bar in
         [0,1]^K with ||p_bar||_1 = L (default sum-of-squares coverage; a
         custom coverage + solver pair can be plugged in via params).
         Madow systematic sampling draws a size-L super action.
      3. After observing per-arm rewards y on the played arms A, take ONE
         plain MSE gradient step on the L-row mini-batch (phi[A], y) — the
         canonical SquareCB regression oracle, matching ``learnOnline`` in
         ``example_code_oe2d``.

    Restricted to RegressionPolicy: per-arm features must be available via
    ``x.get_ld_features()``. Enumeration / classification reductions are
    no longer supported by this class.
    """

    def __init__(self, B, learning_alg=None, classification=False):
        if classification:
            raise NotImplementedError(
                "Online SquareCB supports per-arm regression only."
            )
        self.B = B
        self.learning_alg = learning_alg
        self.policy_type = RegressionPolicy

    def init(self, T, params=None):
        params = params or {}

        self.gamma_multiplier = float(params.get('gamma_multiplier', 1.0))
        self.gamma_schedule = params.get('gamma_schedule', 'sqrt_kt')
        self.coverage = params.get('coverage', default_coverage)
        self.solver = params.get('solver', None)
        self.sampler = params.get('sampler', 'marginal')
        if self.sampler not in ('marginal', 'bandit', 'depround'):
            raise ValueError(f"Unknown sampler: {self.sampler!r}")

        seed = params.get('seed', None)
        self.rng = np.random.default_rng(seed)

        # ``model_type`` is the oracle short name (linear / rff / nonlinear /
        # gb5 / gb2 / xgb / ...). Anything in ``params`` that isn't a
        # SquareCB-level key is forwarded to the oracle factory; the
        # registry filters per-oracle kwargs and each oracle owns its own
        # defaults. Adding a new oracle does not require touching SquareCB.
        self._oracle_name = params.get('model_type', 'linear')
        self._oracle_params = {
            k: v for k, v in params.items() if k not in _SQUARECB_KEYS
        }
        self._oracle_params.setdefault('seed', seed)
        self.model: Optional[OnlineOracle] = None

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.imp_weights = None

    def _build_model(self, ld_dim) -> OnlineOracle:
        if self.model is None:
            self.model = make_oracle(
                self._oracle_name, ld_dim=int(ld_dim), **self._oracle_params
            )
        return self.model

    def _gamma(self) -> float:
        if callable(self.gamma_schedule):
            return float(self.gamma_schedule(self.t, self.B.K, self.B.L))  # type: ignore[arg-type]
        return self.gamma_multiplier * np.sqrt(self.B.K * self.t / self.B.L)

    def _predict_per_arm(self, x) -> np.ndarray:
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        model = self._build_model(features.shape[1])
        return model.predict(features)

    def get_action(self, x):
        L = x.get_L()
        f = self._predict_per_arm(x)
        gamma = self._gamma()
        if self.sampler == 'bandit':
            slate = igw_bandit_sample_slate(f, gamma, L, self.rng)
            self.imp_weights = None
        else:
            p_bar = igw_marginals(f, gamma, L, solver=self.solver)
            if self.sampler == 'depround':
                slate = depround_slate(p_bar, L, self.rng)
            else:  # 'marginal' (Madow systematic)
                slate = sample_slate(p_bar, L, self.rng)
            self.imp_weights = p_bar
        self.action = slate
        return slate

    def update(self, x, A, y_vec, r):
        del r  # Unused: SquareCB updates from per-arm y_vec on the played arms.
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        A = np.asarray(A, dtype=np.int64)
        phi_played = features[A]
        y_played = np.asarray(y_vec, dtype=np.float64)

        model = self._build_model(features.shape[1])
        model.step(phi_played, y_played)

        self.action = None
        self.imp_weights = None
        self.t += 1
