"""
Registry of online per-arm reward oracles for SquareCB.

An oracle implements two methods:

    predict(features: np.ndarray[K, d]) -> np.ndarray[K]      # original-y units
    step(phi_played: np.ndarray[L, d], y_played: np.ndarray[L]) -> float

``make_oracle(name, ld_dim, **params)`` selects the implementation.
Per-oracle keyword filtering keeps SquareCB free of oracle-specific
plumbing — it can hand the union of all oracle params to the registry.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OnlineOracle(Protocol):
    """Per-arm reward predictor with a single online update per round."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def step(self, phi_played: np.ndarray, y_played: np.ndarray) -> float: ...


# Torch (sigmoid head) oracle sub-types share one class.
_TORCH_TYPES = {"linear", "rff", "nonlinear"}

# GB oracle aliases → max_depth. ``gb`` reads ``max_depth`` from kwargs.
_GB_DEPTH = {"gb5": 5, "gb2": 2, "gb": None}

# XGBoost oracle aliases → max_depth. ``xgb`` reads ``max_depth`` from kwargs.
_XGB_DEPTH = {"xgb5": 5, "xgb2": 2, "xgb": None}

_TORCH_KEYS = {
    "initlr", "tzero", "device", "numrff", "rff_sigma", "depth",
    "normalize_features", "y_scale", "seed",
}
_GB_KEYS = {
    "learning_rate", "buffer_size", "fit_every", "max_trees",
    "min_samples_leaf", "warmup",
    "normalize_features", "y_scale", "seed",
}
_XGB_KEYS = {
    "learning_rate", "buffer_size", "fit_every", "num_boost_round",
    "reg_lambda", "reg_alpha", "min_child_weight",
    "subsample", "colsample_bytree", "tree_method", "warmup",
    "normalize_features", "y_scale", "seed",
}


def make_oracle(name: str, ld_dim: int, **params) -> OnlineOracle:
    """Construct an online oracle by short name.

    Unknown kwargs are dropped (SquareCB passes the union of all oracle
    params; each implementation only consumes its own).
    """
    if name in _TORCH_TYPES:
        from .online_regressor import OnlinePerArmRegressor
        kwargs = {k: v for k, v in params.items() if k in _TORCH_KEYS}
        return OnlinePerArmRegressor(ld_dim=int(ld_dim), model_type=name, **kwargs)

    if name in _GB_DEPTH:
        from .online_gb_oracle import OnlineGBOracle
        kwargs = {k: v for k, v in params.items() if k in _GB_KEYS}
        depth = _GB_DEPTH[name]
        if depth is None:
            depth = int(params.get("max_depth", 5))
        return OnlineGBOracle(ld_dim=int(ld_dim), max_depth=int(depth), **kwargs)

    if name in _XGB_DEPTH:
        from .online_xgb_oracle import OnlineXGBOracle
        kwargs = {k: v for k, v in params.items() if k in _XGB_KEYS}
        depth = _XGB_DEPTH[name]
        if depth is None:
            depth = int(params.get("max_depth", 5))
        return OnlineXGBOracle(ld_dim=int(ld_dim), max_depth=int(depth), **kwargs)

    raise ValueError(
        f"Unknown oracle name: {name!r}. "
        f"Known: {sorted(_TORCH_TYPES | _GB_DEPTH.keys() | _XGB_DEPTH.keys())}"
    )
