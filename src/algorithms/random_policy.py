import numpy as np

from .base import Semibandit


class RandomPolicy(Semibandit):
    """
    Uniform-random slate baseline: at each round, pick a size-L subset of
    {1, ..., K} uniformly at random and do no learning.

    Subclassed from Semibandit for the get_action / play loop, but overrides
    init to skip the self.B.N reference (which DatasetBandit does not set).
    """

    def init(self, T, params=None):
        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        pass

    def get_action(self, x):
        K = x.get_K()
        L = x.get_L()
        self.action = np.random.choice(K, size=L, replace=False)
        return self.action
