"""
SquareCB.Lin log-det sampler for L-of-K semi-bandits.

Follows Foster, Gentile, Mohri, Zimmert (2021) "Adapting to Misspecification in Contextual Bandits" (arXiv:2107.05745), Algorithm 2 + Definition 1, specialized to one-hot per-arm features e_i (so d = K) and adapted to the combinatorial slate setting by enumerating super-actions of size L.

Original logdet-barrier objective (Eq. 6 in the paper):

    argmin_{p ∈ Δ(A)} { <ā_p, θ> - γ^{-1} log det(H_p - ā_p ā_p^T) }

Our simplification: drop the ā_p ā_p^T centering, keep only H_p:

    argmin_{p ∈ Δ(A)} { <ā_p, θ> - γ^{-1} log det(H_p) }

In our codebase we operate on per-arm REWARDS f (not losses), so we flip the linear sign:

    p_t = argmin_{p ∈ Δ_N} { -<ā_p, f> - γ^{-1} log det(H_p) }

with
    N = C(K, L), the number of size-L subsets of [K],
    a_n = sum_{i in S_n} e_i in R^K (multi-hot row vector),
    ā_p = sum_n p_n a_n in R^K (per-arm marginals; sum to L),
    H_p = sum_n p_n a_n a_n^T in R^{K x K} (diagonal = marginals, off-diagonal = pairwise co-occurrence probabilities).

The objective is convex (linear + composition of -log det with an affine map).
The default ``'cvxpy'`` solves the log_det model with CLARABEL only,
without a fallback solver. ``'frank_wolfe'`` retains the iterative method.
All backends use H_p + ridge * I and enumerate all C(K, L) super-actions.
Select a backend by name or supply a callable with the _solve_frank_wolfe signature.
"""

import numpy as np
from itertools import combinations
from typing import Callable, Optional, Tuple, Union

SolverFn = Callable[..., np.ndarray]
SolverLike = Union[str, SolverFn, None]

_SUBSET_CACHE: dict = {}
_FEATURE_CACHE: dict = {}

def _enumerate_subsets(K: int, L: int) -> np.ndarray:
    """All size-L subsets of [K] as a length-N x L int64 array, cached."""
    key = (int(K), int(L))
    cached = _SUBSET_CACHE.get(key)
    if cached is None:
        cached = np.array(list(combinations(range(K), L)), dtype=np.int64)
        _SUBSET_CACHE[key] = cached
    return cached


def _enumerate_features(K: int, L: int) -> np.ndarray:
    """Multi-hot feature matrix A_feat of shape (N, K), cached."""
    key = (int(K), int(L))
    cached = _FEATURE_CACHE.get(key)
    if cached is None:
        subsets = _enumerate_subsets(K, L)
        N = subsets.shape[0]
        A_feat = np.zeros((N, K), dtype=np.float64)
        rows = np.repeat(np.arange(N), L)
        cols = subsets.ravel()
        A_feat[rows, cols] = 1.0
        _FEATURE_CACHE[key] = A_feat
        cached = A_feat
    return cached


def _solve_frank_wolfe(
    A_feat: np.ndarray,
    f: np.ndarray,
    gamma: float,
    max_iter: int = 80,
    tol: float = 1e-5,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Frank-Wolfe on Δ_N for J(q) = -<ā_q, f> - γ^{-1} log det(H_q + ridge·I).

    Reward convention: J corresponds to maximizing predicted reward plus a log-det exploration bonus, recast as a convex minimization.

    Per-iteration cost: O(N·K^2 + K^3); for K=10, N=120 each iteration is~10^4 flops, so 80 iterations land at ~10^6 flops/round.
    """
    N, K = A_feat.shape

    q = np.full(N, 1.0 / N, dtype=np.float64)
    if gamma <= 0:
        return q  # γ ≤ 0 disables the linear term; uniform is exact optimum.

    inv_gamma = 1.0 / float(gamma)
    ridge_I = ridge * np.eye(K)

    for it in range(max_iter):
        # H_q = sum_n q[n] a_n a_n^T = A_feat^T diag(q) A_feat
        H = A_feat.T @ (q[:, None] * A_feat) + ridge_I
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # Mildly singular; bump ridge once and continue.
            H_inv = np.linalg.inv(H + 1e-3 * np.eye(K))

        # quad[n] = a_n^T H^{-1} a_n  via einsum; shape (N,)
        quad = np.einsum("nk,kl,nl->n", A_feat, H_inv, A_feat)
        # grad J w.r.t. q_n = -<a_n, f> - (1/γ) a_n^T H^{-1} a_n
        grad = -(A_feat @ f) - inv_gamma * quad

        n_star = int(np.argmin(grad))
        # FW duality gap on the simplex
        gap = float(grad @ q - grad[n_star])
        if gap < tol:
            break

        eta = 2.0 / (it + 2.0)
        q *= 1.0 - eta
        q[n_star] += eta

    # Defensive: clamp tiny negatives from fp drift and renormalize.
    np.maximum(q, 0.0, out=q)
    s = float(q.sum())
    if s > 0 and abs(s - 1.0) > 0:
        q /= s
    return q


def _solve_cvxpy(
    A_feat: np.ndarray,
    f: np.ndarray,
    gamma: float,
    max_iter: int = 80,
    tol: float = 1e-8,
    ridge: float = 1e-6,
    *,
    cvxpy_solver: Optional[str] = None,
    **solve_kwargs,
) -> np.ndarray:
    """Solve the same ridge-regularized log-det program directly with CVXPY.

    Use CLARABEL only and propagate solver failures without a retry. max_iter
    caps solver iterations and tol sets feasibility/gap tolerances. The default
    max_step_fraction=0.95 avoids observed stalled interior-point steps. Explicit
    solve_kwargs override the CLARABEL defaults. The optimal q need not
    be unique; compare objectives, not individual weights.

    For positive ridge, remove null directions of A_feat before taking log_det.
    Their contribution is the constant (K - rank) * log(ridge), so this keeps
    the optimizer while avoiding a poorly conditioned, larger PSD cone.
    """
    if cvxpy_solver not in (None, "CLARABEL"):
        raise ValueError("The CVXPY backend supports only CLARABEL")
    if solve_kwargs.pop("solver", "CLARABEL") != "CLARABEL":
        raise ValueError("The CVXPY backend supports only CLARABEL")
    A_feat = np.asarray(A_feat, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N, K = A_feat.shape
    if gamma <= 0 or N == 1:
        return np.full(N, 1.0 / N, dtype=np.float64)

    rewards = A_feat @ f
    if np.isposinf(gamma):
        # The exploration term vanishes, leaving a linear simplex problem.
        q = (rewards == np.max(rewards)).astype(np.float64)
        return q / q.sum()

    import cvxpy as cp

    moment_features = A_feat
    if ridge > 0:
        _, singular_values, right_vectors = np.linalg.svd(A_feat, full_matrices=False)
        cutoff = max(A_feat.shape) * np.finfo(np.float64).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > cutoff))
        if rank == 0:
            return np.full(N, 1.0 / N, dtype=np.float64)
        if rank < K:
            moment_features = A_feat @ right_vectors[:rank].T
    dimension = moment_features.shape[1]
    q = cp.Variable(N, nonneg=True, name="q")
    # Row scaling avoids constructing an N x N diagonal matrix.
    H = moment_features.T @ cp.multiply(cp.reshape(q, (N, 1), order="C"), moment_features)
    H = H + ridge * np.eye(dimension)
    # Subtracting a constant reward on the simplex preserves the optimizer.
    rewards = rewards - rewards.max()
    problem = cp.Problem(
        cp.Minimize(-(rewards @ q) - cp.log_det(H) / float(gamma)),
        [cp.sum(q) == 1.0],
    )
    defaults = {
        "max_iter": max_iter, "tol_gap_abs": tol,
        "tol_gap_rel": tol, "tol_feas": tol,
        "max_step_fraction": 0.95,
    }
    for key, value in defaults.items():
        solve_kwargs.setdefault(key, value)
    problem.solve(solver="CLARABEL", **solve_kwargs)
    if q.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            "cvxpy failed on the log-det program "
            f"(status={problem.status!r}, gamma={gamma}, N={N}, K={K})"
        )

    # Clip negative drift up to max(tol, 0.01), then check the sum and normalize.
    result = np.asarray(q.value, dtype=np.float64).ravel()
    if not np.all(np.isfinite(result)) or np.min(result) < -max(tol, 1e-2):
        raise RuntimeError("cvxpy returned an invalid log-det distribution")
    result = np.maximum(result, 0.0)
    total = float(result.sum())
    if total <= 0.0 or abs(total - 1.0) > max(10 * tol, 1e-1):
        raise RuntimeError("cvxpy returned an invalid log-det distribution sum")
    return result / total


LOGDET_SOLVERS: dict = {
    "frank_wolfe": _solve_frank_wolfe,
    "cvxpy": _solve_cvxpy,
}
DEFAULT_LOGDET_SOLVER = "cvxpy"


def _solve_logdet(
    A_feat: np.ndarray,
    f: np.ndarray,
    gamma: float,
    max_iter: int = 80,
    tol: float = 1e-8,
    ridge: float = 1e-6,
    *,
    solver: SolverLike = None,
) -> np.ndarray:
    """Dispatch a log-det solve, retaining the original positional arguments.

    None selects DEFAULT_LOGDET_SOLVER. A callable receives (A_feat, f, gamma, max_iter=..., tol=..., ridge=...) and returns q of shape (N,).
    """
    if solver is None:
        solver = DEFAULT_LOGDET_SOLVER
    if not callable(solver):
        key = str(solver).strip().lower()
        if key not in LOGDET_SOLVERS:
            raise ValueError(
                f"unknown log-det solver {solver!r}; expected one of "
                f"{sorted(LOGDET_SOLVERS)} or a callable"
            )
        solver = LOGDET_SOLVERS[key]
    q = np.asarray(
        solver(A_feat, f, gamma, max_iter=max_iter, tol=tol, ridge=ridge),
        dtype=np.float64,
    )
    if (q.shape != (A_feat.shape[0],) or not np.all(np.isfinite(q))
            or np.any(q < 0.0) or abs(float(q.sum()) - 1.0) > 1e-8):
        raise RuntimeError("log-det solver must return a finite nonnegative distribution of shape (N,) summing to 1")
    return q


def logdet_sample_slate(
    f: np.ndarray,
    gamma: float,
    L: int,
    rng: np.random.Generator,
    max_iter: int = 80,
    tol: float = 1e-8,
    ridge: float = 1e-6,
    *,
    solver: SolverLike = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sample a size-L slate via the simplified logdet-barrier program.

    solver selects 'cvxpy' (default, CLARABEL only), 'frank_wolfe', or a compatible callable. Existing positional arguments and the returned tuple are unchanged.

    Returns (slate, q, n_sampled):
        slate      -- length-L int64 array of arm indices.
        q          -- length-N float64 super-action distribution (Σq = 1).
        n_sampled  -- index in [0, N) of the sampled super-action.
    """
    f = np.asarray(f, dtype=np.float64)
    K = int(f.shape[0])
    L = int(L)
    if not (1 <= L <= K):
        raise ValueError(f"L={L} out of range [1, {K}]")

    subsets = _enumerate_subsets(K, L)
    A_feat = _enumerate_features(K, L)
    N = subsets.shape[0]

    if L == K:
        # Single super-action, no choice to make.
        return subsets[0].copy(), np.ones(1, dtype=np.float64), 0

    q = _solve_logdet(
        A_feat, f, gamma, max_iter=max_iter, tol=tol, ridge=ridge, solver=solver
    )

    # Categorical draw; numpy needs an exact-sum-to-1 vector, so we already renormalized inside _solve_logdet.
    n_sampled = int(rng.choice(N, p=q))
    slate = subsets[n_sampled].copy()
    return slate, q, n_sampled
