import numpy as np

from Policy import EnumerationPolicy, ClassificationPolicy, RegressionPolicy
import Argmax

from .base import Semibandit
from .igw_sampler import default_coverage, igw_marginals, sample_slate


class SquareCB(Semibandit):
    """
    SquareCB / IGW for the L-of-K semi-bandit setting.

    At each round, a regression oracle f_hat predicts per-item rewards.
    The IGW sampler turns those scores into marginal inclusion
    probabilities p_bar in [0,1]^K with ||p_bar||_1 = L (default
    sum-of-squares coverage; a custom coverage + solver pair can be
    plugged in via params). A super action of size L is then drawn by
    Madow systematic sampling so that item-inclusion marginals match
    p_bar exactly. Importance weights are 1 / p_bar(k) on the played
    items, mirroring MiniMonster's bookkeeping.

    The regression oracle is refit on an exponential schedule using the
    same Argmax.argmax2 path as MiniMonster / EpsGreedy.
    """

    def __init__(self, B, learning_alg=None, classification=False):
        self.B = B
        self.learning_alg = learning_alg
        if learning_alg is None:
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

    def init(self, T, params=None):
        params = params or {}

        self.gamma_multiplier = float(params.get('gamma_multiplier', 1.0))
        self.gamma_schedule = params.get('gamma_schedule', 'sqrt_kt')
        self.coverage = params.get('coverage', default_coverage)
        self.solver = params.get('solver', None)

        seed = params.get('seed', None)
        self.rng = np.random.default_rng(seed)

        # Exponential training schedule, same as MiniMonster.
        self.training_points = []
        i = 4
        while True:
            self.training_points.append(int(np.sqrt(2) ** i))
            i += 1
            if np.sqrt(2) ** i > T:
                break
        if params.get('schedule') == 'lin':
            self.training_points = [50]
            j = 1
            while True:
                self.training_points.append(int(j * 100))
                j += 1
                if j * 100 > T:
                    break

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.imp_weights = None
        self.f_hat = None
        self.history = []
        self.num_unif = 0

    def _gamma(self) -> float:
        if callable(self.gamma_schedule):
            return float(self.gamma_schedule(self.t, self.B.K, self.B.L))
        if self.gamma_schedule == 'const':
            return self.gamma_multiplier
        # Default: gamma_multiplier * sqrt(K * t / L), the canonical
        # SquareCB scaling adapted to L-of-K semi-bandits.
        return self.gamma_multiplier * np.sqrt(self.B.K * self.t / self.B.L)

    def _predict_per_arm(self, x) -> np.ndarray:
        """Per-base-item reward estimates from the current regression oracle."""
        if self.policy_type is RegressionPolicy:
            features = np.asarray(x.get_ld_features())
            preds = self.f_hat.model.predict(features)
            return np.reshape(np.asarray(preds, dtype=np.float64), x.get_K())
        # Enumeration / classification: oracle returns a top-L slate
        # rather than per-item scores, so synthesize a score vector that
        # ranks the chosen slate at the top.
        slate = self.f_hat.get_action(x)
        scores = np.zeros(x.get_K(), dtype=np.float64)
        scores[slate] = 1.0
        return scores

    def get_action(self, x):
        K = x.get_K()
        L = x.get_L()
        if self.f_hat is None:
            slate = np.random.choice(K, size=L, replace=False)
            self.imp_weights = (L / K) * np.ones(K)
            self.action = slate
            self.num_unif += 1
            return slate

        f = self._predict_per_arm(x)
        gamma = self._gamma()
        p_bar = igw_marginals(f, gamma, L, solver=self.solver)
        slate = sample_slate(p_bar, L, self.rng)
        self.imp_weights = p_bar
        self.action = slate
        return slate

    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec
        # 1 / p_bar on the played items; on the rest, the entry is unused
        # by the regression oracle but we mirror MiniMonster's storage layout.
        self.history.append((x, A, full_rvec, 1.0 / np.clip(self.imp_weights, 1e-12, None)))

        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("---- Training ----", flush=True)
            self.f_hat = Argmax.argmax2(
                self.B,
                self.history,
                policy_type=self.policy_type,
                learning_alg=self.learning_alg,
            )

        self.action = None
        self.imp_weights = None
        self.t += 1
