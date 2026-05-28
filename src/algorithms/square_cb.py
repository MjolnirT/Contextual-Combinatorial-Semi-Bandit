import numpy as np

from Policy import RegressionPolicy

from typing import Optional

from .base import Semibandit
from .igw_sampler import (depround_slate, igw_bandit_sample_slate,
                          igw_marginals, sample_slate)
from .logbarrier_sampler import (
    _enumerate_features as _logbarrier_features,
    logbarrier_sample_slate,
)
from .online_oracles import OnlineOracle, make_oracle


class SquareCB(Semibandit):
    """
    SquareCB / IGW for the L-of-K semi-bandit setting with online
    per-arm regression updates.

    Each round:
      1. Predict per-item rewards f_hat(phi_k) for all K arms via the
         online regressor.
      2. IGW turns scores into marginal inclusion probabilities p_bar in
         [0,1]^K with ||p_bar||_1 = L by maximizing
         ``<p, f_hat> + (1/gamma) * sum_k log(p_k)`` (the log-barrier
         exploration program; see ``igw_sampler.igw_marginals``). A
         custom solver can be plugged in via ``params['solver']``.
         Madow systematic sampling draws a size-L super action.
      3. After observing per-arm rewards y on the played arms A, take ONE
         plain MSE gradient step on the L-row mini-batch (phi[A], y) — the
         canonical SquareCB regression oracle, matching ``learnOnline`` in
         ``example_code_oe2d``.

    Restricted to RegressionPolicy: per-arm features must be available via
    ``x.get_ld_features()``.
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
        params = dict(params or {})

        self.gamma_multiplier = float(params.get('gamma_multiplier', 1.0))
        self.gamma_schedule = params.get('gamma_schedule', 'sqrt_kt')
        self.solver = params.get('solver', None)
        self.sampler = params.get('sampler', 'marginal')
        if self.sampler not in ('marginal', 'bandit', 'depround', 'logbarrier'):
            raise ValueError(f"Unknown sampler: {self.sampler!r}")

        seed = params.get('seed', None)
        self.rng = np.random.default_rng(seed)

        # Forward everything to the oracle factory; make_oracle filters
        # per-oracle, so SquareCB-only / legacy keys are dropped silently.
        self._oracle_name = params.get('model_type', 'linear')
        params.setdefault('seed', seed)
        self._oracle_params = params
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

    def get_action(self, x):
        L = x.get_L()
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        f = self._build_model(features.shape[1]).predict(features)
        gamma = self._gamma()
        if self.sampler == 'bandit':
            slate = igw_bandit_sample_slate(f, gamma, L, self.rng)
            self.imp_weights = None
        elif self.sampler == 'logbarrier':
            slate, q, _ = logbarrier_sample_slate(f, gamma, L, self.rng)
            # Store per-arm marginals for shape consistency with the IGW
            # marginal sampler (downstream PV / logging treats imp_weights
            # as a length-K vector).
            self.imp_weights = q @ _logbarrier_features(int(f.shape[0]), L)
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

        self._build_model(features.shape[1]).step(phi_played, y_played)

        self.action = None
        self.imp_weights = None
        self.t += 1

    def predict_slate_value(self, x, p):
        if self.model is None:
            return float('nan')
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        preds = self.model.predict(features[np.asarray(p), :])
        return float(np.sum(preds))
