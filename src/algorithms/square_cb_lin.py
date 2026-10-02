"""
SquareCB.lin for the L-of-K setting, under BANDIT feedback.

Foster, Gentile, Mohri, Zimmert (2021) "Adapting to Misspecification in Contextual Bandits" (arXiv:2107.05745), Algorithm 2: a contextual LINEAR bandit over super-actions. Two design differences from ``square_cb_comb.SquareCB``:

  1. Action selection: the log-det-barrier program over all C(K, L) super-actions (``sampling_oracle.logdet_sampler.logdet_sample_slate``) instead of per-arm IGW marginals + Madow.
  2. Feedback: BANDIT, not semi-bandit. ``update`` gives the oracle ONE regression sample per round — the sum of the played arms' features (the super-action vector a = sum_{k in A} phi_k) paired with the scalar slate reward — and treats the per-arm rewards as unobserved. This is the feedback model the log-det exploration is built for: the barrier keeps the played covariance H_q well-conditioned, which is what makes per-arm means identifiable from aggregate rewards.

The oracle interface is shared with SquareCB (``learning_oracles.online_oracles.make_oracle``). The linear oracle is the faithful choice — FGMZ's algorithm is linear by construction; nonlinear oracles are accepted but make the row a labeled hybrid rather than the paper's algorithm.
"""

import numpy as np

from Policy import RegressionPolicy
from typing import Optional
from .base import Semibandit
from .sampling_oracle.logdet_sampler import (
    _enumerate_features as _logdet_features,
    logdet_sample_slate,
)
from .learning_oracles.online_oracles import OnlineOracle, make_oracle


class SquareCBLin(Semibandit):
    """
    SquareCB.lin: log-det-barrier super-action sampling + bandit-feedback online regression.

    Each round:
      1. Predict per-item rewards f_hat(phi_k) for all K arms via the online regressor; a super-action's predicted reward is the sum over its arms (linear in the summed features).
      2. Solve the log-det program over the C(K, L) super-action simplex (Frank-Wolfe; see ``logdet_sampler``) and sample a slate from the resulting distribution q.
      3. Observe ONLY the slate reward r; take one MSE gradient step on the single sample (sum_{k in A} phi_k, r).
    """

    def __init__(self, B, learning_alg=None, classification=False):
        if classification:
            raise NotImplementedError(
                "SquareCB.lin supports per-arm regression oracles only."
            )
        self.B = B
        self.learning_alg = learning_alg
        self.policy_type = RegressionPolicy

    def init(self, T, params=None):
        params = dict(params or {})
        params.pop('sampler', None)  # CLI compatibility; this class IS the logdet sampler

        self.gamma_multiplier = float(params.get('gamma_multiplier', 1.0))
        self.gamma_schedule = params.get('gamma_schedule', 'sqrt_kt')

        seed = params.get('seed', None)
        self.rng = np.random.default_rng(seed)

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
        slate, q, _ = logdet_sample_slate(f, self._gamma(), L, self.rng)
        # Per-arm marginals of q, for logging parity with SquareCB.
        self.imp_weights = q @ _logdet_features(int(f.shape[0]), L)
        self.action = slate
        return slate

    def update(self, x, A, y_vec, r):
        del y_vec  # bandit feedback: per-arm rewards are treated as unobserved
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        phi = features[np.asarray(A, dtype=np.int64)].sum(axis=0, keepdims=True)
        self._build_model(features.shape[1]).step(phi, np.array([float(np.sum(r))]))
        self.action = None
        self.imp_weights = None
        self.t += 1
