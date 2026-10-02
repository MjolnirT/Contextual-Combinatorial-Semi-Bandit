import numpy as np

from .base import Semibandit


class RandomPolicy(Semibandit):
    """
    Uniform-random slate baseline: at each round, pick a size-L subset of {1, ..., K} uniformly at random and do no learning.

    Subclassed from Semibandit for the get_action / play loop, but overrides init to skip the self.B.N reference (which DatasetBandit does not set).

    Slate selection uses a per-run seeded RNG (``params['seed']``) so the reward trace is reproducible for a given seed and differs across seeds.
    """

    def init(self, T, params=None):
        params = dict(params or {})
        self.rng = np.random.default_rng(params.get('seed', None))
        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        pass

    def get_action(self, x):
        K = x.get_K()
        L = x.get_L()
        self.action = self.rng.choice(K, size=L, replace=False)
        return self.action
