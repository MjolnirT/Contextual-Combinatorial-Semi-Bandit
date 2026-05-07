"""Cached Gram-matrix OLS for MiniMonster's inner-loop AMO.

Inside `MiniMonster._solve_op`, the AMO is called up to ~20 times per
training point on a dataset with two structural properties:

  1. The same X matrix is used in every call (history features), and
  2. X grows append-only across training points.

`GramCache` keeps `X^T X` (with intercept column) updated incrementally.
The actual feature matrix is held by the caller and passed into `solve`.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class GramCache:
    """Append-only X^T X cache with intercept column, indexed by history index.

    Storage is just (d+1) x (d+1); the feature rows themselves live with the
    caller.
    """

    JITTER = 1e-8

    def __init__(self, d):
        self.d = d
        self.dim = d + 1
        self._XtX = np.zeros((self.dim, self.dim))
        self._n_rows = 0
        self._row_ranges = []
        self._cho = None
        self._cho_at_n = -1

    def n_history(self):
        return len(self._row_ranges)

    def n_rows(self):
        return self._n_rows

    def append(self, features):
        """Fold the K rows for the next history index into X^T X.

        Returns the row range (lo, hi) so callers can index into their own
        feature buffer at the same offsets.
        """
        feats = np.asarray(features)
        K = feats.shape[0]
        ## XtX block = [F^T F, F^T 1; 1^T F, K]
        FtF = feats.T @ feats
        col_sum = feats.sum(axis=0)
        d = self.d
        self._XtX[:d, :d] += FtF
        self._XtX[:d, d] += col_sum
        self._XtX[d, :d] += col_sum
        self._XtX[d, d] += K
        lo = self._n_rows
        self._n_rows += K
        hi = self._n_rows
        self._row_ranges.append((lo, hi))
        self._cho = None
        return (lo, hi)

    def row_range(self, i):
        return self._row_ranges[i]

    def solve(self, X, y):
        """Solve `(X^T X) β = X^T y`. Returns (coef, intercept).

        `X` is the (n_rows, d) feature matrix held by the caller (without
        intercept column); `y` has length n_rows. The Cholesky factor is
        reused as long as no rows have been appended since the last call.
        """
        n = self._n_rows
        assert y.shape[0] == n, f"y has length {y.shape[0]}, expected {n}"
        assert X.shape == (n, self.d), \
            f"X has shape {X.shape}, expected ({n}, {self.d})"
        if self._cho is None or self._cho_at_n != n:
            G = self._XtX.copy()
            G[np.arange(self.dim), np.arange(self.dim)] += self.JITTER
            self._cho = cho_factor(G, lower=True)
            self._cho_at_n = n
        rhs = np.empty(self.dim)
        rhs[:self.d] = X.T @ y
        rhs[self.d] = y.sum()
        beta = cho_solve(self._cho, rhs)
        return beta[:-1].copy(), float(beta[-1])


class LinModel:
    """sklearn-compatible model wrapping (coef, intercept).

    Used by `RegressionPolicy.get_all_actions` which only needs `.predict`.
    """

    __slots__ = ("coef_", "intercept_")

    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_
