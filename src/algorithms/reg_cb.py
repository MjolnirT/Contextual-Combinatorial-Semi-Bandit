"""
RegCB — direct port of ``RegressorUCB`` from ``example_code_regcb/Semibandits.py`` (Dylan Foster's 2017 prototype on the ``regression`` branch of akshaykr/oracle_cb), which became RegCB in Foster, Agarwal, Dudík, Luo, Schapire, ICML 2018, "Practical Contextual Bandits with Regression Oracles".

Restricted to the contextual bandit setting (L=1).

Pull strategies (from the prototype's ``STRATEGIES``):
  * ``greedy``            ≡ paper's RegCB-opt   (``argmax_a UCB(x,a)``)
  * ``pull_if_uncertain`` ≡ paper's RegCB-elim  (uniform over ``{a : UCB(a) >= max_a' LCB(a')}``)

Oracles (incremental regressors with a ``pred_range`` confidence band):
  * ``lin``  → ``Incremental.IncrementalLinearRegression(reg=1.)``
  * ``gb5``  → ``Incremental.IncrementalRegressionTreeEnsemble2(max_depth=5, n_estimators=100)``

The runner (``run_semibandits.py``) passes ``--param`` as ``radius`` (the empirical-least-squares-regret slack β in the paper) and ``--regcb_mode`` as the pull strategy. All other RegressorUCB defaults (``burn_in=200``, ``cheat=True``, ``stack_actions=False``, ``prec=0.001``) are kept verbatim.
"""

import numpy as np
import sys
import os

from .base import Semibandit
from .learning_oracles import incremental as Incremental
from Policy import RegressionPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegCB(Semibandit):
    """Foster's ``RegressorUCB`` prototype, ported verbatim."""

    STRATEGIES = ['greedy', 'pull_if_uncertain']

    def __init__(self, B, learning_alg=None, classification=False):
        """
        Args:
            B           -- Semibandit Simulator (must have L==1).
            learning_alg -- one of {'lin', 'gb5'}, or a callable factory returning an Incremental regressor instance.
            classification -- not supported (kept for runner compatibility).
        """
        if classification:
            raise NotImplementedError('RegCB does not support classification reduction.')
        self.B = B
        if self.B.L != 1:
            raise NotImplementedError('RegCB requires L=1 (contextual bandit).')
        self._learning_alg_arg = learning_alg

    # ------------------------------------------------------------------ #
    def _resolve_oracle_factory(self, name):
        if callable(name):
            return name
        if name == 'lin':
            return lambda: Incremental.IncrementalLinearRegression(reg=1.)
        if name == 'gb5':
            return lambda: Incremental.IncrementalRegressionTreeEnsemble2(
                max_depth=5, n_estimators=100)
        raise ValueError(
            f"RegCB supports learning_alg in {{'lin','gb5'}} or a factory; got {name!r}"
        )

    # ------------------------------------------------------------------ #
    def init(self, T, params=None):
        """
        Initialize the run. Maps runner params onto the original RegressorUCB knobs:
          --param / 'beta'      → self.radius
          --regcb_mode          → self.pull_strategy
                                    'optimistic'  → 'greedy'
                                    'elimination' → 'pull_if_uncertain'
          --learning_alg / 'learning_alg_name' → oracle factory
        """
        params = params or {}

        # Algorithm parameters --- (mirrors RegressorUCB.init lines 149-169)
        self.stack_actions = bool(params.get('regcb_stack_actions', False))
        self.cheat = bool(params.get('regcb_cheat', True))

        mode = params.get('regcb_mode', 'optimistic')
        if mode == 'optimistic':
            self.pull_strategy = 'greedy'
        elif mode == 'elimination':
            self.pull_strategy = 'pull_if_uncertain'
        elif mode in self.STRATEGIES:
            self.pull_strategy = mode
        else:
            raise ValueError(f"Unknown regcb_mode: {mode!r}")

        self.T = T
        self.burn_in = int(params.get('regcb_burn_in', 200))

        # ``radius`` is the β in the paper; runner surfaces it as --param.
        self.radius = float(params.get('beta', params.get('param', 0.01)))
        self.prec = float(params.get('regcb_prec', 0.001))

        # Confidence parameters (kept verbatim; mostly unused).
        self.beta = 0.1
        self.failure_prob = 0.05
        self.nu = np.log(2 * T ** 2 * self.B.K / self.failure_prob)
        self.kappa = 80
        self.eps = self.T ** (self.beta) * self.nu
        self.delta = 1

        # Resolve oracle factory (string from runner OR callable from caller).
        name = params.get('learning_alg_name', self._learning_alg_arg)
        self.learning_alg = self._resolve_oracle_factory(name)

        self.t = 1

        # Training schedule: burn_in + 2^k
        self.training_points = []
        self.training_indices = []
        t = 1
        while t + self.burn_in <= T:
            self.training_points.append(self.burn_in + t)
            t *= 2

        self.leaders = None
        self.leader = None
        self.uncertain_t = None

        self.subset_sizes = None
        self.upper_range = None
        self.lower_range = None

        self.reward = []
        self.opt_reward = []
        self.history = []
        self.action = None

        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(int(seed))

    # ------------------------------------------------------------------ #
    def update_confidence(self, i):
        self.eta = 1. / np.sqrt(i)
        self.delta = self.kappa * self.eps / (float(i) - 1)
        self.eps = (self.T / float(i)) ** (self.beta) * self.nu

    # ------------------------------------------------------------------ #
    def update(self, x, act, y_vec, r):
        if self.uncertain_t is not None and self.uncertain_t == self.t:
            full_rvec = np.zeros(self.B.K)
            full_rvec[act] = y_vec
            self.history.append((x, act, full_rvec, np.ones(self.B.K)))

        if self.t in self.training_points:
            print("training. (t={})".format(self.t))

            m = len(self.history)

            if self.stack_actions is True:
                Xs = [np.zeros((m, self.B.d)) for _ in range(self.B.K)]
                Ys = [np.zeros(m) for _ in range(self.B.K)]
                subset_sizes = [0 for _ in range(self.B.K)]

                for item in self.history:
                    context = item[0]
                    a = item[1][0]
                    reward = item[2]
                    ss = subset_sizes[a]
                    Xs[a][ss] = context.get_ld_features()[a, :]
                    Ys[a][ss] = reward[a]
                    subset_sizes[a] += 1

                self.subset_sizes = subset_sizes
                self.leaders = []
                for idx in range(self.B.K):
                    Xs[idx] = Xs[idx][:subset_sizes[idx]]
                    Ys[idx] = Ys[idx][:subset_sizes[idx]]
                    pred = self.learning_alg()
                    pred.fit(Xs[idx], Ys[idx])
                    self.leaders.append(RegressionPolicy(pred))
            else:
                X = np.zeros((m, self.B.d))
                Y = np.zeros(m)
                for (idx, item) in enumerate(self.history):
                    context = item[0]
                    a = item[1][0]
                    reward = item[2]
                    X[idx] = context.get_ld_features()[a, :]
                    Y[idx] = reward[a]
                pred = self.learning_alg()
                pred.fit(X, Y)
                self.leader = RegressionPolicy(pred)

        self.training_indices.append(len(self.history) - 1)
        self.t += 1
        self.update_confidence(self.t)

    # ------------------------------------------------------------------ #
    def get_action(self, x):
        if self.t <= self.training_points[0]:
            act = np.random.choice(self.B.K, size=1, replace=False)
            self.imp_weights = np.ones(self.B.K) / float(self.B.K)
            self.uncertain_t = self.t
            self.action = act
            return act

        upper_range = np.zeros(self.B.K)
        lower_range = np.zeros(self.B.K)

        for idx in range(self.B.K):
            xa = x.get_ld_features()[idx, :]

            if self.stack_actions is True:
                leader = self.leaders[idx]
            else:
                leader = self.leader

            model = leader.model
            m = model.get_dataset_size()

            if self.cheat is False:
                (rmin, rmax) = model.pred_range_coarse(xa, self.radius)
                lmin = 0.
                lmax = m + self.prec
                leader_mse = model.get_mse()
                r_upper = self._binary_search(model, xa, self.radius, rmax, self.prec, lmin, lmax, leader_mse)
                r_lower = self._binary_search(model, xa, self.radius, rmin, self.prec, lmin, lmax, leader_mse)
            else:
                (r_lower, r_upper) = model.pred_range(xa, self.prec)

            upper_range[idx] = r_upper
            lower_range[idx] = r_lower

        self.upper_range = upper_range
        self.lower_range = lower_range

        if self.pull_strategy == 'pull_if_uncertain':
            max_lower = np.max(lower_range)
            possible_winners = [idx for idx in range(self.B.K)
                                if upper_range[idx] >= max_lower]
            assert len(possible_winners) > 0
            uncertain = (len(possible_winners) > 1)
            if uncertain:
                self.uncertain_t = self.t
                act = np.array([possible_winners[np.random.choice(len(possible_winners))]])
            else:
                act = np.array([possible_winners[0]])
        elif self.pull_strategy == 'greedy':
            act = np.array([np.argmax(upper_range)])
            self.uncertain_t = self.t
        else:
            raise ValueError(f"Unknown pull_strategy: {self.pull_strategy!r}")

        self.action = act
        return self.action

    # ------------------------------------------------------------------ #
    def _binary_search(self, model, x, radius, r, prec, lmin, lmax, min_mse):
        """
        Binary search over the inverse Lagrange multiplier ``1/lt`` for the constrained least-squares problem of (Foster et al. 2018) Algorithm 3.

        Args:
            r -- target reward value (rmin or rmax) we try to match.
            model -- IncrementalRegressionModel (supports fit_incremental).
        """
        ll = lmin
        lh = lmax
        lt = None

        while lh - ll > prec:
            lt = (ll + lh) / 2.
            (pred, past_mse, full_mse) = model.fit_incremental(x, r, weight=1. / lt, keep=False)
            if max(past_mse - min_mse, 0) > radius:
                ll = lt
            else:
                lh = lt

        if lt is None:
            return model.predict(x)
        return pred
