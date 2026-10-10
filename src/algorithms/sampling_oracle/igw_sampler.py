"""
IGW sampler for L-of-K semi-bandits.

Given estimated per-item rewards f_hat (shape K), an exploration parameter gamma, and a slate size L, ``igw_marginals`` returns a marginal inclusion distribution p_bar in [0,1]^K with ||p_bar||_1 = L. ``sample_slate_madow`` then draws a size-L super action whose item-inclusion marginals match p_bar exactly (Madow systematic sampling).

``igw_marginals`` solves a log-barrier exploration program:

    max_p   <p, f_hat>  +  (1/gamma) * sum_k log(p_k)
    s.t.    sum_k p_k = L,    0 < p_k <= 1

The program itself is solved by a pluggable back-end, selected through the ``solver`` argument of ``igw_marginals``:

  * ``None`` (default) / ``'closed_form'`` -- ``_solve_closed_form``: the analytic KKT solution, a 1-D root find (``scipy.optimize.brentq``) inside a saturation loop. No dependency beyond scipy; this is what every experiment in the repo runs.
  * ``'cvxpy'`` -- ``_solve_cvxpy``: hands the same program to CVXPY as a generic convex program. Reference / cross-check path only: CVXPY is an OPTIONAL dependency (imported lazily, so the module still imports without it) and is orders of magnitude slower per call than the closed form.
  * any callable ``solver(f_hat, gamma, L) -> p_bar``, as before.

Back-ends are registered in ``IGW_SOLVERS``; both built-ins are exact drop-in callables (they short-circuit the same degenerate inputs), so they may also be called directly.
"""

import numpy as np

from typing import Callable, Optional, Union
from scipy.optimize import brentq


# solver(f_hat, gamma, L) -> p_bar
SolverFn = Callable[[np.ndarray, float, int], np.ndarray]
SolverLike = Union[str, SolverFn, None]


def igw_marginals(
    f_hat: np.ndarray,
    gamma: float,
    L: int,
    solver: SolverLike = None,
) -> np.ndarray:
    """
    Marginal inclusion probabilities p_bar in [0,1]^K with sum = L from the log-barrier exploration program

        max_p   <p, f_hat>  +  (1/gamma) * sum_k log(p_k)
        s.t.    sum_k p_k = L,    0 < p_k <= 1.

    ``solver`` picks the back-end that solves it: ``None`` (default) or ``'closed_form'`` for the analytic KKT solve, ``'cvxpy'`` for the CVXPY cross-check, or any callable ``(f_hat, gamma, L) -> p_bar``. Whatever the back-end, the returned marginals are checked by ``_validate_marginals``.
    """
    if solver is None:
        solver = DEFAULT_IGW_SOLVER
    if not callable(solver):
        key = str(solver).strip().lower()
        if key not in IGW_SOLVERS:
            raise ValueError(
                "unknown IGW solver %r; expected one of %s or a callable "
                "(f_hat, gamma, L) -> p_bar" % (solver, sorted(IGW_SOLVERS))
            )
        solver = IGW_SOLVERS[key]
    p = np.asarray(solver(f_hat, gamma, L), dtype=np.float64)
    _validate_marginals(p, L)
    return p


def _degenerate_marginals(f_hat: np.ndarray, gamma: float,
                          L: int) -> Optional[np.ndarray]:
    """
    The solver-independent cases, or None when the real program must be solved.

    Shared by every built-in back-end so they agree exactly on the boundary and so no back-end is ever handed an input outside the log barrier's domain (gamma = 0 -> infinite barrier weight; L = K -> the feasible set is the single point p = 1; gamma = +inf -> no barrier at all).
    """
    K = f_hat.shape[0]
    if not (1 <= L <= K):
        raise ValueError("L must satisfy 1 <= L <= K, got L=%d, K=%d" % (L, K))
    if L == K:
        return np.ones(K, dtype=np.float64)
    if gamma <= 0:
        # No exploitation pressure -> log barrier dominates -> uniform L/K.
        return np.full(K, L / K, dtype=np.float64)

    coef = 1.0 / float(gamma)
    if not np.isfinite(coef) or coef <= 0:
        # gamma -> +inf: barrier vanishes -> top-L pure exploit.
        p = np.zeros(K, dtype=np.float64)
        top = np.argpartition(f_hat, -L)[-L:]
        p[top] = 1.0
        return p
    return None


def _solve_closed_form(f_hat: np.ndarray, gamma: float, L: int) -> np.ndarray:
    """
    Analytic KKT solve (the default back-end).

    With gap_k = f_max - f_hat_k (>= 0), the KKT optimum for uncapped items is p_k = 1 / (gamma * (u + gap_k)) where u = lambda - f_max > 0 solves the 1-D equation

        sum_k 1/(u + gap_k) = gamma * L_rem.

    h(u) = LHS - RHS is strictly decreasing on (0, +inf) with h(0+) = +inf (top-of-remaining item has gap=0 -> 1/u) and h(+inf) = -gamma * L_rem < 0, so brentq finds the unique root. Items whose p_k would exceed 1 are pinned to 1 and the remainder re-solved (saturation loop terminates because each pass pins at least one new item).
    """
    f_hat = np.asarray(f_hat, dtype=np.float64)
    K = f_hat.shape[0]
    degenerate = _degenerate_marginals(f_hat, gamma, L)
    if degenerate is not None:
        return degenerate

    gamma = float(gamma)
    saturated = np.zeros(K, dtype=bool)
    p = np.zeros(K, dtype=np.float64)

    while True:
        idx = np.where(~saturated)[0]
        L_rem = L - int(saturated.sum())
        n_rem = idx.shape[0]

        if L_rem <= 0:
            p[:] = saturated.astype(np.float64)
            break
        if n_rem <= L_rem:
            p[:] = 0.0
            p[saturated] = 1.0
            p[idx] = 1.0
            break

        f_idx = f_hat[idx]
        f_max = float(np.max(f_idx))
        gap = f_max - f_idx  # >= 0; top-of-remaining items have gap 0

        target = gamma * L_rem

        def h(u: float) -> float:
            return float(np.sum(1.0 / (u + gap))) - target

        # h(u) <= n_rem/u so any u_hi >= n_rem/target gives h(u_hi) <= 0; double for fp safety until the sign flip is confirmed.
        u_lo = 1e-18
        u_hi = max(1.0, n_rem / target)
        while h(u_hi) > 0.0 and u_hi < 1e30:
            u_hi *= 2.0

        u_root = float(brentq(h, u_lo, u_hi, xtol=1e-14))  # type: ignore[arg-type]
        p_idx = 1.0 / (gamma * (u_root + gap))

        over = p_idx > 1.0
        if np.any(over):
            saturated[idx[over]] = True
            continue

        p[saturated] = 1.0
        p[idx] = p_idx
        break

    return p


def _solve_cvxpy(f_hat: np.ndarray, gamma: float, L: int,
                 cvxpy_solver: Optional[str] = None,
                 **solve_kwargs) -> np.ndarray:
    """
    CVXPY back-end: the same program handed to a generic convex solver.

    ``cvxpy_solver`` forwards a solver name (e.g. 'CLARABEL', 'SCS', 'ECOS'), defaulting to CLARABEL when installed -- CVXPY's own pick for this exponential-cone program. Since the point of this back-end is to check the closed form, it also tightens that solver's tolerances (``_CVXPY_TIGHT_DEFAULTS``): at stock tolerances the two agree only to ~1e-5, tightened to ~1e-10, at no measurable cost. Anything passed in ``solve_kwargs`` wins over those defaults. Interior-point output still only satisfies the constraints to solver tolerance, so it is repaired onto the exact box/sum before ``_validate_marginals`` sees it.

    Requesting 1e-12 is deliberately more than CLARABEL can always deliver on the near-degenerate barrier (small gamma * a nearly-saturated coordinate), so a minority of calls stop at ``OPTIMAL_INACCURATE`` and CVXPY emits "Solution may be inaccurate". Those answers are still the most accurate of any tolerance setting measured (~1e-6 vs ~5e-5 at 1e-10), which is why the status is accepted rather than raised on.
    """
    f_hat = np.asarray(f_hat, dtype=np.float64)
    degenerate = _degenerate_marginals(f_hat, gamma, L)
    if degenerate is not None:
        return degenerate

    try:
        import cvxpy as cp
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "the 'cvxpy' IGW solver needs the optional cvxpy package "
            "(pip install cvxpy). The default solver ('closed_form') has no "
            "dependency beyond scipy."
        ) from exc

    K = f_hat.shape[0]
    gamma = float(gamma)
    p = cp.Variable(K, name="p_bar")
    problem = cp.Problem(
        cp.Maximize(gamma * (f_hat @ p) + cp.sum(cp.log(p))),
        [cp.sum(p) == float(L), p <= 1.0],
    )
    if cvxpy_solver is None and "CLARABEL" in cp.installed_solvers():
        cvxpy_solver = "CLARABEL"
    if cvxpy_solver is not None:
        solve_kwargs["solver"] = cvxpy_solver
        for key, val in _CVXPY_TIGHT_DEFAULTS.get(cvxpy_solver, {}).items():
            solve_kwargs.setdefault(key, val)
    problem.solve(**solve_kwargs)

    if p.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            "cvxpy failed on the IGW program (status=%r, gamma=%g, L=%d, K=%d)"
            % (problem.status, gamma, L, K)
        )
    return _repair_marginals(np.asarray(p.value, dtype=np.float64).ravel(), L)


# Per-solver tolerance overrides for ``_solve_cvxpy`` -- see its docstring. A solver missing from this map just runs at its own defaults.
_CVXPY_TIGHT_DEFAULTS = {
    "CLARABEL": {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12},
    "ECOS": {"abstol": 1e-12, "reltol": 1e-12, "feastol": 1e-12},
    "SCS": {"eps": 1e-10, "max_iters": 50000},
}


def _repair_marginals(p: np.ndarray, L: int) -> np.ndarray:
    """
    Snap an approximately-feasible p onto {0 <= p <= 1, sum p = L} exactly.

    Clip into the box, then push the residual L - sum(p) onto the coordinates that still have room, re-clipping each pass. Each pass either kills the residual or saturates at least one more coordinate, so the loop is bounded by K. Only meant to absorb solver-tolerance drift (~1e-9), not to fix a genuinely infeasible point -- ``_validate_marginals`` still has the last word.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    for _ in range(p.shape[0] + 1):
        residual = float(L) - float(p.sum())
        if abs(residual) <= 1e-15 * max(L, 1):
            break
        room = p < 1.0 if residual > 0 else p > 0.0
        n_room = int(room.sum())
        if n_room == 0:
            break
        p[room] = np.clip(p[room] + residual / n_room, 0.0, 1.0)
    return p


IGW_SOLVERS: dict = {
    "closed_form": _solve_closed_form,
    "cvxpy": _solve_cvxpy,
}
DEFAULT_IGW_SOLVER = "closed_form"


def sample_slate_madow(p_bar: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Madow systematic sampling: returns a length-L array of distinct base actions whose inclusion marginals equal ``p_bar``. Items are randomly permuted before the systematic sweep so the joint distribution does not depend on the input ordering.
    """
    p = np.asarray(p_bar, dtype=np.float64)
    K = p.shape[0]
    if not (1 <= L <= K):
        raise ValueError("L must satisfy 1 <= L <= K, got L=%d, K=%d" % (L, K))

    order = rng.permutation(K)
    p_perm = p[order]
    # Cumulative sum with leading zero so bucket k is [cum[k], cum[k+1]).
    cum = np.empty(K + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(p_perm, out=cum[1:])

    # Floating point: pull the upper edge in slightly so searchsorted lands in the last bucket for thresholds near L.
    total = cum[-1]
    if not np.isfinite(total) or abs(total - L) > 1e-6 * max(L, 1):
        raise ValueError("p_bar must sum to L (got %.6f vs %d)" % (total, L))

    u = float(rng.random())
    thresholds = u + np.arange(L, dtype=np.float64)
    # searchsorted with side='right' returns insertion index in cum; subtract 1 to get the bucket index.
    bucket = np.searchsorted(cum, thresholds, side="right") - 1
    bucket = np.clip(bucket, 0, K - 1)

    selected = order[bucket]
    # Madow with width-<=1 buckets cannot duplicate, but defensively assert.
    if np.unique(selected).shape[0] != L:
        raise RuntimeError("Madow produced duplicate selections; check p_bar")
    return selected.astype(np.int64)


def sample_slate_depround(p_bar: np.ndarray, L: int,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Dependent rounding (Gandhi-Khuller-Parthasarathy-Srinivasan 2002).

    Converts a marginal distribution ``p_bar in [0,1]^K`` with ``sum(p_bar) == L`` into a 0/1 slate of size exactly L such that each item k is included with probability exactly ``p_bar[k]``.

    Pair-update rule: while two coordinates p_i, p_j are both in (0, 1), let alpha = min(1 - p_i, p_j) and beta = min(p_i, 1 - p_j); with probability beta/(alpha+beta) take the "alpha move" (p_i += alpha, p_j -= alpha), else the "beta move" (p_i -= beta, p_j += beta). Each iteration drives at least one of {p_i, p_j} to {0, 1}, so the loop terminates in <= K iterations. Item-level marginals and the total sum are preserved exactly in expectation; under floating-point arithmetic the sum is preserved to within ~1e-12, and any residual drift is snapped at the end via top-L selection.

    Compared to Madow systematic sampling (``sample_slate_madow``):
      * Same marginal guarantee (P[k in S] = p_bar[k]).
      * Different joint law -- DepRound has weaker negative correlation between distinct items, but is still negatively associated.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(p_bar, dtype=np.float64).copy()
    K = p.shape[0]
    _validate_marginals(p, L)
    L = int(L)

    EPS = 1e-12
    fractional = np.where((p > EPS) & (p < 1 - EPS))[0].tolist()
    iters = 0
    while len(fractional) >= 2 and iters < 4 * K:
        # Pop two fractional indices; update them; push back any that remain fractional.
        i = fractional.pop()
        j = fractional.pop()
        alpha = min(1.0 - p[i], p[j])
        beta = min(p[i], 1.0 - p[j])
        if alpha + beta <= 0:
            # Already integral due to fp drift; just continue.
            iters += 1
            continue
        if rng.random() < beta / (alpha + beta):
            p[i] += alpha
            p[j] -= alpha
        else:
            p[i] -= beta
            p[j] += beta
        # Re-check fractional status of i and j.
        for k in (i, j):
            if EPS < p[k] < 1.0 - EPS:
                fractional.append(int(k))
        iters += 1

    selected = np.where(p > 0.5)[0]
    if selected.shape[0] != L:
        # Floating-point drift: pick the L largest entries.
        selected = np.argsort(-p, kind='stable')[:L]
    return np.sort(selected).astype(np.int64)


def _validate_marginals(p: np.ndarray, L: int) -> None:
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        raise RuntimeError("p_bar out of [0,1]: min=%.6e, max=%.6e" % (p.min(), p.max()))
    s = float(p.sum())
    if abs(s - L) > 1e-6 * max(L, 1):
        raise RuntimeError("p_bar does not sum to L: sum=%.6f, L=%d" % (s, L))
