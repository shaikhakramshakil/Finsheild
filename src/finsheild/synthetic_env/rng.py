"""Deterministic RNG helpers for the synthetic environment.

We do NOT want pandas/numpy global state to leak across tables. Each table
generator therefore takes an explicit ``numpy.random.Generator`` (built
from a salted sub-stream of the master seed) plus a Python ``random.Random``
instance for categorical draws.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def make_rng(master_seed: int, salt: str) -> np.random.Generator:
    """Return a numpy generator seeded deterministically from the master seed.

    ``salt`` distinguishes sub-streams so two tables never share the same RNG
    state.
    """
    digest = hashlib.sha256(f"{master_seed}:{salt}".encode()).digest()
    sub_seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(sub_seed)


def make_random(master_seed: int, salt: str) -> random.Random:
    """Return a Python ``random.Random`` seeded from the same scheme."""
    digest = hashlib.sha256(f"{master_seed}:{salt}".encode()).digest()
    sub_seed = int.from_bytes(digest[8:16], "big", signed=False)
    return random.Random(sub_seed)