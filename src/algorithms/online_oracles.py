"""
Registry of online per-arm reward oracles for SquareCB.

An oracle implements two methods:

    predict(features: np.ndarray[K, d]) -> np.ndarray[K]      # original-y units
    step(phi_played: np.ndarray[L, d], y_played: np.ndarray[L]) -> float

``make_oracle(name, ld_dim, **params)`` selects the implementation. The
factory introspects each oracle's __init__ signature, so SquareCB can
forward its full params dict and unrelated keys are dropped silently.
"""

import inspect
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OnlineOracle(Protocol):
    """Per-arm reward predictor with a single online update per round."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def step(self, phi_played: np.ndarray, y_played: np.ndarray) -> float: ...


# ``gb`` / ``dt`` / ``xgb`` (no number) fall back to max_depth in kwargs
# (default 5).
_DEPTH_ALIASES = {
    "gb2": 2, "gb5": 5,
    "dt2": 2, "dt5": 5,
    "xgb2": 2, "xgb5": 5,
}


def _kwargs_for(cls, params, *, drop=()):
    """Keep params accepted by ``cls.__init__``, minus any in ``drop``
    (used to avoid clashes with kwargs the caller passes explicitly)."""
    accepted = set(inspect.signature(cls).parameters) - {"self", *drop}
    return {k: v for k, v in params.items() if k in accepted}


def make_oracle(name: str, ld_dim: int, **params) -> OnlineOracle:
    if name in {"linear", "rff", "nonlinear"}:
        from .online_regressor import OnlinePerArmRegressor
        return OnlinePerArmRegressor(
            ld_dim=int(ld_dim), model_type=name,
            **_kwargs_for(OnlinePerArmRegressor, params,
                          drop=("ld_dim", "model_type")),
        )

    if name in {"gb", "gb2", "gb5"}:
        from .online_gb_oracle import OnlineGBOracle
        depth = _DEPTH_ALIASES.get(name, int(params.get("max_depth", 5)))
        return OnlineGBOracle(
            ld_dim=int(ld_dim), max_depth=int(depth),
            **_kwargs_for(OnlineGBOracle, params,
                          drop=("ld_dim", "max_depth")),
        )

    if name in {"dt", "dt2", "dt5"}:
        from .online_dt_oracle import OnlineDTOracle
        depth = _DEPTH_ALIASES.get(name, int(params.get("max_depth", 5)))
        return OnlineDTOracle(
            ld_dim=int(ld_dim), max_depth=int(depth),
            **_kwargs_for(OnlineDTOracle, params,
                          drop=("ld_dim", "max_depth")),
        )

    if name in {"xgb2", "xgb5"}:
        from .online_xgb_oracle import OnlineXGBOracle
        depth = _DEPTH_ALIASES[name]
        return OnlineXGBOracle(
            ld_dim=int(ld_dim), max_depth=int(depth),
            **_kwargs_for(OnlineXGBOracle, params,
                          drop=("ld_dim", "max_depth")),
        )

    raise ValueError(f"Unknown oracle name: {name!r}")
