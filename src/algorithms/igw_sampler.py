"""
IGW sampler for L-of-K semi-bandits.

Given estimated per-item rewards f_hat (shape K), an exploration parameter
gamma, and a slate size L, ``igw_marginals`` returns a marginal inclusion
distribution p_bar in [0,1]^K with ||p_bar||_1 = L. ``sample_slate`` then
draws a size-L super action whose item-inclusion marginals match p_bar
exactly (Madow systematic sampling).

``igw_marginals`` solves a log-barrier exploration program:

    max_p   <p, f_hat>  +  (1/gamma) * sum_k log(p_k)
    s.t.    sum_k p_k = L,    0 < p_k <= 1

Larger gamma -> smaller log-barrier weight -> sharper p (more exploit),
matching the standard SquareCB direction where gamma_t grows with t.
KKT (ignoring the upper cap) yields p_k = (1/gamma) / (lambda - f_hat_k)
with lambda > max_k f_hat_k chosen so sum_k p_k = L; items whose p_k
would exceed 1 are pinned to 1 and the remainder re-solved.

Empirical motivation: head-to-head against the legacy sum-of-squares
coverage IGW (closed-form 1/(n + gamma*gap - z) saturation solve), the
log-barrier program produced strictly better cumulative reward across
{linear, gb2, xgb2} oracles and gamma_multiplier in {0.1, 1, 10} on the
xor benchmark; the SS solver was removed.

A user can short-circuit the closed-form solve by supplying a custom
``solver(f_hat, gamma, L) -> p_bar``.
"""

from math import comb
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq


def igw_marginals(
    f_hat: np.ndarray,
    gamma: float,
    L: int,
    solver: Optional[Callable[[np.ndarray, float, int], np.ndarray]] = None,
) -> np.ndarray:
    """
    Marginal inclusion probabilities p_bar in [0,1]^K with sum = L from
    the log-barrier exploration program

        max_p   <p, f_hat>  +  (1/gamma) * sum_k log(p_k)
        s.t.    sum_k p_k = L,    0 < p_k <= 1.

    With coef = 1/gamma and gap_k = f_max - f_hat_k (>= 0), the KKT
    optimum is p_k = coef / (u + gap_k) where u = lambda - f_max > 0
    solves the 1-D equation

        sum_k 1/(u + gap_k) = L_rem / coef.

    h(u) = LHS - RHS is strictly decreasing on (0, +inf) with h(0+) = +inf
    (top-of-remaining item has gap=0 -> 1/u) and h(+inf) = -L_rem/coef < 0,
    so brentq finds the unique root. Items whose p_k would exceed 1 are
    pinned to 1 and the remainder re-solved (saturation loop terminates
    because each pass pins at least one new item).
    """
    if solver is not None:
        p = np.asarray(solver(f_hat, gamma, L), dtype=np.float64)
        _validate_marginals(p, L)
        return p

    f_hat = np.asarray(f_hat, dtype=np.float64)
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

        target = L_rem / coef

        def h(u: float) -> float:
            return float(np.sum(1.0 / (u + gap))) - target

        # h(u) <= n_rem/u so any u_hi >= n_rem/target gives h(u_hi) <= 0;
        # double for fp safety until the sign flip is confirmed.
        u_lo = 1e-18
        u_hi = max(1.0, n_rem / target)
        while h(u_hi) > 0.0 and u_hi < 1e30:
            u_hi *= 2.0

        u_root = float(brentq(h, u_lo, u_hi, xtol=1e-14))  # type: ignore[arg-type]
        p_idx = coef / (u_root + gap)

        over = p_idx > 1.0
        if np.any(over):
            saturated[idx[over]] = True
            continue

        p[saturated] = 1.0
        p[idx] = p_idx
        break

    _validate_marginals(p, L)
    return p


def sample_slate(p_bar: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Madow systematic sampling: returns a length-L array of distinct base
    actions whose inclusion marginals equal ``p_bar``. Items are randomly
    permuted before the systematic sweep so the joint distribution does
    not depend on the input ordering.
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

    # Floating point: pull the upper edge in slightly so searchsorted lands
    # in the last bucket for thresholds near L.
    total = cum[-1]
    if not np.isfinite(total) or abs(total - L) > 1e-6 * max(L, 1):
        raise ValueError("p_bar must sum to L (got %.6f vs %d)" % (total, L))

    u = float(rng.random())
    thresholds = u + np.arange(L, dtype=np.float64)
    # searchsorted with side='right' returns insertion index in cum;
    # subtract 1 to get the bucket index.
    bucket = np.searchsorted(cum, thresholds, side="right") - 1
    bucket = np.clip(bucket, 0, K - 1)

    selected = order[bucket]
    # Madow with width-<=1 buckets cannot duplicate, but defensively assert.
    if np.unique(selected).shape[0] != L:
        raise RuntimeError("Madow produced duplicate selections; check p_bar")
    return selected.astype(np.int64)


def depround_slate(p_bar: np.ndarray, L: int,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Dependent rounding (Gandhi-Khuller-Parthasarathy-Srinivasan 2002).

    Converts a marginal distribution ``p_bar in [0,1]^K`` with
    ``sum(p_bar) == L`` into a 0/1 slate of size exactly L such that each
    item k is included with probability exactly ``p_bar[k]``.

    Pair-update rule: while two coordinates p_i, p_j are both in (0, 1),
    let alpha = min(1 - p_i, p_j) and beta = min(p_i, 1 - p_j); with
    probability beta/(alpha+beta) take the "alpha move" (p_i += alpha,
    p_j -= alpha), else the "beta move" (p_i -= beta, p_j += beta).
    Each iteration drives at least one of {p_i, p_j} to {0, 1}, so the
    loop terminates in <= K iterations. Item-level marginals and the
    total sum are preserved exactly in expectation; under floating-point
    arithmetic the sum is preserved to within ~1e-12, and any residual
    drift is snapped at the end via top-L selection.

    Compared to Madow systematic sampling (``sample_slate``):
      * Same marginal guarantee (P[k in S] = p_bar[k]).
      * Different joint law -- DepRound has weaker negative correlation
        between distinct items, but is still negatively associated.
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
        # Pop two fractional indices; update them; push back any that
        # remain fractional.
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


def igw_bandit_sample_slate(
    scores: np.ndarray,
    gamma: float,
    L: int,
    rng: Optional[np.random.Generator] = None,
    max_retries: int = 10000,
) -> np.ndarray:
    """
    Sample a size-L slate by treating each L-subset of [K] as a single
    macro-action and applying scalar SquareCB IGW over them. Sampling
    proceeds via standard rejection.

    Macro-action scoring is additive: f(S) = sum_{k in S} scores[k].
    With M = C(K, L) and gap(S) = f_max - f(S) (>= 0), IGW assigns

        p(S) = 1 / (M + gamma * gap(S))     for S != S*
        p(S*) = 1 - sum_{S != S*} p(S)

    Rejection sampler:
        propose S ~ Uniform(C(K, L))
        accept with probability  M / (M + gamma * gap(S))   in [0, 1]
        otherwise resample

    The argmax slate S* has gap=0 and is always accepted, bounding the
    expected loop length by O(M); for typical gamma the accept rate
    stays close to 1 and the loop terminates in a few iterations. A
    safety ``max_retries`` falls back to S* to guarantee termination.

    Parameters
    ----------
    scores : array_like, shape (K,)
        Per-arm reward predictions f_hat(k).
    gamma : float
        IGW exploration parameter.
    L : int
        Slate size, 1 <= L <= K.
    rng : np.random.Generator, optional
    max_retries : int
        Hard cap on rejection iterations before falling back to S*.

    Returns
    -------
    slate : np.ndarray of shape (L,), dtype int64
        Sorted indices of the sampled macro-action.
    """
    if rng is None:
        rng = np.random.default_rng()
    f = np.asarray(scores, dtype=np.float64)
    K = int(f.shape[0])
    if not (1 <= L <= K):
        raise ValueError(f"L={L} out of range [1, {K}]")

    top_L = np.argpartition(f, -L)[-L:]
    f_max = float(f[top_L].sum())
    M = float(comb(K, L))
    g = float(gamma)

    for _ in range(max_retries):
        a = rng.choice(K, size=L, replace=False)
        gap = f_max - float(f[a].sum())
        # gap can be a tiny negative due to fp; clamp to >= 0.
        if gap < 0.0:
            gap = 0.0
        accept_prob = M / (M + g * gap)
        if rng.random() < accept_prob:
            return np.sort(a).astype(np.int64)
    # Fallback: argmax slate (always accepted under IGW).
    return np.sort(top_L).astype(np.int64)
