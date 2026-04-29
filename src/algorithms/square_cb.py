import numpy as np

from Policy import RegressionPolicy

from .base import Semibandit
from .igw_sampler import (default_coverage, igw_bandit_sample_slate,
                          igw_marginals, sample_slate)
from .online_regressor import OnlinePerArmRegressor


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
        if self.sampler not in ('marginal', 'bandit'):
            raise ValueError(f"Unknown sampler: {self.sampler!r}")

        seed = params.get('seed', None)
        self.rng = np.random.default_rng(seed)

        self._regressor_params = {
            'model_type': params.get('model_type', 'linear'),
            'initlr': float(params.get('initlr', 0.05)),
            'tzero': float(params.get('tzero', 100.0)),
            'device': params.get('device', 'cpu'),
            'numrff': int(params.get('numrff', 512)),
            'rff_sigma': float(params.get('rff_sigma', 1.0)),
            'depth': int(params.get('depth', 2)),
            'seed': seed,
        }
        self.model = None

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.imp_weights = None

    def _build_model(self, ld_dim):
        if self.model is None:
            self.model = OnlinePerArmRegressor(
                ld_dim=int(ld_dim), **self._regressor_params
            )

    def _gamma(self) -> float:
        if callable(self.gamma_schedule):
            return float(self.gamma_schedule(self.t, self.B.K, self.B.L))
        return self.gamma_multiplier * np.sqrt(self.B.K * self.t / self.B.L)

    def _predict_per_arm(self, x) -> np.ndarray:
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        self._build_model(features.shape[1])
        return self.model.predict(features)

    def get_action(self, x):
        L = x.get_L()
        f = self._predict_per_arm(x)
        gamma = self._gamma()
        if self.sampler == 'bandit':
            slate = igw_bandit_sample_slate(f, gamma, L, self.rng)
            self.imp_weights = None
        else:
            p_bar = igw_marginals(f, gamma, L, solver=self.solver)
            slate = sample_slate(p_bar, L, self.rng)
            self.imp_weights = p_bar
        self.action = slate
        return slate

    def update(self, x, A, y_vec, r):
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        A = np.asarray(A, dtype=np.int64)
        phi_played = features[A]
        y_played = np.asarray(y_vec, dtype=np.float64)

        self.model.step(phi_played, y_played)

        self.action = None
        self.imp_weights = None
        self.t += 1
