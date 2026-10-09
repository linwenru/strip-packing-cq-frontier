"""Data models for the bidirectional best-fit heuristic (BBF)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

from burke_bf.model import Placement, Policy


class VerticalExact(str, Enum):
    """Policy#1: exact fit into the vertical niche (Table 1)."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"


class HorizontalExact(str, Enum):
    """Policy#2: rectangle selection for exact fit into the horizontal niche."""

    TRE = "TRE"
    NRE = "NRE"


class ExactOrder(str, Enum):
    """Policy#3: which exact-fit search runs first."""

    EHV = "eHV"
    EVH = "eVH"


class HorizontalBestFit(str, Enum):
    """Policy#4: rectangle selection for best fit into the horizontal gap."""

    BP = "BP"
    FP = "FP"


class VerticalBestFit(str, Enum):
    """Policy#5: rectangle selection for best fit into the vertical niche."""

    FH = "FH"
    WR = "WR"
    NO_VB = "noVB"


class BestFitOrder(str, Enum):
    """Policy#6: which best-fit search runs first."""

    BHV = "bHV"
    BVH = "bVH"


@dataclass(frozen=True, slots=True)
class Combination:
    """One of the 288 policy combinations tested exhaustively by the paper."""

    vertical_exact: VerticalExact
    horizontal_exact: HorizontalExact
    exact_order: ExactOrder
    horizontal_best: HorizontalBestFit
    vertical_best: VerticalBestFit
    best_order: BestFitOrder
    placement: Policy

    @property
    def value(self) -> str:
        # Matches the paper's ordered 7-tuple notation, e.g. Table 2.
        return (
            f"{self.vertical_exact.value},{self.horizontal_exact.value},"
            f"{self.exact_order.value},{self.horizontal_best.value},"
            f"{self.vertical_best.value},{self.best_order.value},"
            f"{self.placement.short_name}"
        )


def all_combinations() -> list[Combination]:
    """Enumerate every policy combination in the paper's nesting order."""

    return [
        Combination(*parts)
        for parts in product(
            VerticalExact,
            HorizontalExact,
            ExactOrder,
            HorizontalBestFit,
            VerticalBestFit,
            BestFitOrder,
            Policy,
        )
    ]


@dataclass(frozen=True, slots=True)
class Solution:
    """A complete packing produced by one policy combination.

    The ``policy`` property aliases the combination so that burke_bf's
    duck-typed helpers (render_svg, validate_solution) work unmodified.
    """

    instance_name: str
    combination: Combination
    height: int
    placements: tuple[Placement, ...]
    skyline: tuple[int, ...]

    @property
    def policy(self) -> Combination:
        return self.combination
