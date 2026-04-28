"""
Semibandit learning algorithms.

Each algorithm lives in its own module; this package exposes them
together so callers can write `from algorithms import EpsGreedy` etc.
"""
from .base import Semibandit
from .eps_greedy import EpsGreedy
from .lin_ucb import LinUCB
from .eels import EELS
from .eels2 import EELS2
from .mini_monster import MiniMonster
from .semi_exp4 import SemiExp4

__all__ = [
    "Semibandit",
    "EpsGreedy",
    "LinUCB",
    "EELS",
    "EELS2",
    "MiniMonster",
    "SemiExp4",
]
