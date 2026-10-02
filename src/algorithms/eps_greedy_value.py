import numpy as np

from Policy import RegressionPolicy
from typing import Optional
from .base import Semibandit
from .learning_oracles.online_oracles import OnlineOracle, make_oracle


class EpsGreedyValue(Semibandit):
    """
    Epsilon-greedy over slates with an online per-arm regression oracle.

    Each round:
      1. Predict per-arm rewards f_hat(phi_k) for all K arms via the online regressor (same oracle interface as ``SquareCB``).
      2. With probability epsilon, play a uniformly random size-L slate (pure exploration over the candidate slate set). Otherwise, play the greedy slate: the L arms with the highest predicted reward.
      3. After observing per-arm rewards y on the played arms A, take one MSE gradient step on (phi[A], y) — same update as ``SquareCB``.

    Restricted to ``RegressionPolicy``: per-arm features must be available via ``x.get_ld_features()``.
    """

    def __init__(self, B, learning_alg=None, classification=False):
        if classification:
            raise NotImplementedError(
                "EpsGreedyValue supports per-arm regression only."
            )
        self.B = B
        self.learning_alg = learning_alg
        self.policy_type = RegressionPolicy

    def init(self, T, params=None):
        params = dict(params or {})

        self.eps = float(params.get('eps', 0.1))

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
        self.ber = 0
        self.num_unif = 0

    def _build_model(self, ld_dim) -> OnlineOracle:
        if self.model is None:
            self.model = make_oracle(
                self._oracle_name, ld_dim=int(ld_dim), **self._oracle_params
            )
        return self.model

    def _greedy_slate(self, f, L):
        return np.argpartition(-f, L - 1)[:L].astype(np.int64)

    def _uniform_slate(self, K, L):
        return self.rng.choice(K, size=L, replace=False).astype(np.int64)

    def get_action(self, x):
        K = x.get_K()
        L = x.get_L()
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        f = self._build_model(features.shape[1]).predict(features)

        self.ber = int(self.rng.random() < self.eps)
        if self.ber:
            slate = self._uniform_slate(K, L)
            self.num_unif += 1
        else:
            slate = self._greedy_slate(f, L)
        self.action = slate
        return slate

    def update(self, x, A, y_vec, r):
        del r
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        A = np.asarray(A, dtype=np.int64)
        phi_played = features[A]
        y_played = np.asarray(y_vec, dtype=np.float64)

        self._build_model(features.shape[1]).step(phi_played, y_played)

        self.action = None
        self.t += 1