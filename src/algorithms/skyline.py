import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import xgboost as xgb

from Policy import RegressionPolicy
from .base import Semibandit


def _make_batch_regressor(oracle_name):
    """Factory for the batch (full-data) supervised regressor used by the skyline. Mirrors the oracle families in run_semibandits.py's make_learning_alg, but fit once offline on the whole training set rather than online.
    """
    if oracle_name == 'lin':
        return sklearn.linear_model.LinearRegression()
    if oracle_name in ('gb2', 'gb5'):
        depth = 2 if oracle_name == 'gb2' else 5
        return sklearn.ensemble.GradientBoostingRegressor(
            n_estimators=100, max_depth=depth)
    if oracle_name in ('xgb2', 'xgb5'):
        depth = 2 if oracle_name == 'xgb2' else 5
        return xgb.XGBRegressor(n_estimators=100, max_depth=depth)
    raise ValueError(f"Skyline does not support oracle {oracle_name!r}; "
                     f"expected one of lin, gb2, gb5, xgb2, xgb5.")


class SkylinePolicy(Semibandit):
    """
    Supervised-learning skyline (in-sample upper bound).

    Before the run, a regression oracle is fit *once* on the entire training set: every (arm feature row, true relevance) pair across all queries. During the run the fully-trained oracle predicts per-arm reward and the policy greedily plays the top-L arms. No exploration, no online update — the oracle has already seen all the data, so this is a ceiling on what a supervised per-arm ranker can achieve on the training stream.

    Restricted to ``RegressionPolicy`` (per-arm features via ``x.get_ld_features()``). Deterministic given the fit.
    """

    def __init__(self, B, learning_alg=None, classification=False):
        if classification:
            raise NotImplementedError(
                "SkylinePolicy supports per-arm regression only.")
        self.B = B
        self.learning_alg = learning_alg
        self.policy_type = RegressionPolicy
        self.model = None

    def _fit_full_data(self):
        """Batch-fit the oracle on all (arm, relevance) pairs in the train set. ``B.contexts.features`` is [X, K, d] and ``relevances`` is [X, K]; flatten both to a single (X*K, d) regression problem."""
        feats = np.asarray(self.B.contexts.features, dtype=np.float64)
        rels = np.asarray(self.B.contexts.relevances, dtype=np.float64)
        X, K, d = feats.shape
        Xmat = np.nan_to_num(feats.reshape(X * K, d))
        yvec = rels.reshape(X * K)
        model = _make_batch_regressor(self.learning_alg)
        if self.verbose:
            print(f"Skyline: batch-fitting {self.learning_alg} on "
                  f"{X * K} (arm, relevance) rows (X={X}, K={K}, d={d})",
                  flush=True)
        model.fit(Xmat, yvec)
        return model

    def init(self, T, params=None):
        if self.learning_alg is None:
            raise ValueError("SkylinePolicy requires a learning_alg.")
        self.reward = []
        self.opt_reward = []
        self.model = self._fit_full_data()

    def update(self, x, A, y_vec, r):
        pass

    def _greedy_slate(self, f, L):
        return np.argpartition(-f, L - 1)[:L].astype(np.int64)

    def get_action(self, x):
        L = x.get_L()
        features = np.asarray(x.get_ld_features(), dtype=np.float64)
        f = self.model.predict(np.nan_to_num(features))
        self.action = self._greedy_slate(np.asarray(f, dtype=np.float64), L)
        return self.action
