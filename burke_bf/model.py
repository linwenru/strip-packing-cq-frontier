"""Data models for rectangular strip packing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Policy(str, Enum):
    """The three niche-placement policies evaluated by the paper."""

    LEFTMOST = "leftmost"
    TALLEST_NEIGHBOUR = "tallest-neighbour"
    SHORTEST_NEIGHBOUR = "shortest-neighbour"

    @property
    def short_name(self) -> str:
        return {
            Policy.LEFTMOST: "LM",
            Policy.TALLEST_NEIGHBOUR: "TN",
            Policy.SHORTEST_NEIGHBOUR: "SN",
        }[self]


@dataclass(frozen=True, slots=True)
class Item:
    """One physical rectangular item."""

    item_id: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Instance:
    """A fixed-width, open-height strip-packing instance."""

    name: str
    strip_width: int
    reference_height: int | None
    items: tuple[Item, ...]

    @property
    def total_area(self) -> int:
        return sum(item.area for item in self.items)


@dataclass(frozen=True, slots=True)
class Placement:
    """A positioned and oriented rectangle."""

    item_id: str
    x: int
    y: int
    width: int
    height: int
    original_width: int
    original_height: int

    @property
    def top(self) -> int:
        return self.y + self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def rotated(self) -> bool:
        return (
            self.width == self.original_height
            and self.height == self.original_width
            and self.original_width != self.original_height
        )


@dataclass(frozen=True, slots=True)
class Solution:
    """A complete packing produced by one placement policy."""

    instance_name: str
    policy: Policy
    height: int
    initial_height: int
    placements: tuple[Placement, ...]
    skyline: tuple[int, ...]
    tower_moves: int
