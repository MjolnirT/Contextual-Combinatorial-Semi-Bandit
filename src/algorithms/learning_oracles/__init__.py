"""Regression and classification oracles used by semibandit algorithms."""

from .online_oracles import OnlineOracle, make_oracle


__all__ = ["OnlineOracle", "make_oracle"]
