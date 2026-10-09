"""Data models for the three-way best-fit heuristic (TWBF)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

from burke_bf.model import Placement


class Ordering(str, Enum):
    """The three input orderings of the three-way best-fit heuristic."""

    WIDTH = "width"
    HEIGHT = "height"
    SURFACE = "surface"


class TwPolicy(str, Enum):
    """The six placement policies: three original plus three new."""

    LEFTMOST = "LM"
    RIGHTMOST = "RM"
    TALLEST = "TN"
    SHORTEST = "SN"
    MIN_DIFF = "MinD"
    MAX_DIFF = "MaxD"


@dataclass(frozen=True, slots=True)
class Combination:
    """One of the 18 ordering x placement-policy combinations."""

    ordering: Ordering
    policy: TwPolicy

    @property
    def value(self) -> str:
        return f"{self.ordering.value}-{self.policy.value}"


def all_combinations() -> list[Combination]:
    return [Combination(*parts) for parts in product(Ordering, TwPolicy)]


@dataclass(frozen=True, slots=True)
class Solution:
    """A complete packing produced by one combination.

    The ``policy`` property aliases the combination so that burke_bf's
    duck-typed helpers (render_svg, validate_solution) work unmodified.
    """

    instance_name: str
    combination: Combination
    height: int
    initial_height: int
    placements: tuple[Placement, ...]
    skyline: tuple[int, ...]
    tower_moves: int

    @property
    def policy(self) -> Combination:
        return self.combination
