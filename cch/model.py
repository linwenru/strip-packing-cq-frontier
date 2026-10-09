"""Component model for the configurable heuristic engine (CCH)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Ordering(str, Enum):
    """D2: input orderings (all decreasing)."""

    WIDTH = "width"
    HEIGHT = "height"
    AREA = "area"
    PERIMETER = "perimeter"
    MAXSIDE = "maxside"
    DIAGONAL = "diagonal"


class Selection(str, Enum):
    """D3: rectangle selection rules for the lowest-gap (BF) frame.

    TRE/NRE fall back to first-fit when no exact niche match exists; the
    niche-only form could otherwise strand a flat full-width gap, which
    cannot be raised (both neighbours are sheet sides).
    """

    WIDEST_FIT = "widest-fit"
    FIRST_FIT = "first-fit"
    TRE = "tre"
    NRE = "nre"
    FITNESS_NUMBER = "fitness-number"
    MAX_AREA = "max-area"


class PlacementPolicy(str, Enum):
    """D4: placement policies within a gap."""

    LM = "LM"
    RM = "RM"
    TN = "TN"
    SN = "SN"
    MIN_DIFF = "MinD"
    MAX_DIFF = "MaxD"


@dataclass(frozen=True, slots=True)
class Config:
    """One point in the component space (a constructive heuristic)."""

    ordering: Ordering = Ordering.WIDTH
    selection: Selection = Selection.WIDEST_FIT
    placement: PlacementPolicy = PlacementPolicy.LM
    tower_removal: bool = False
    rotation_rule: bool = False
    rotatable: bool = True  # RF subtype; ISH/level family use fixed orientation
    # D1: add BBF's vertical niche (region above the leftmost skyline segment
    # below the expected best height) as an alternative placement site.
    vertical_niche: bool = False

    @property
    def value(self) -> str:
        flags = "".join(
            part
            for part, on in (
                ("+tw", self.tower_removal),
                ("+rot", self.rotation_rule),
                ("+of", not self.rotatable),
                ("+vn", self.vertical_niche),
            )
            if on
        )
        return f"{self.ordering.value}/{self.selection.value}/{self.placement.value}{flags}"
