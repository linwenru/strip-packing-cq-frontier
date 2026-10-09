"""Data models for Wei et al.'s skyline heuristic."""

from __future__ import annotations

from dataclasses import dataclass

from burke_bf.model import Placement


@dataclass(frozen=True, slots=True)
class Solution:
    """A complete packing produced by the skyline heuristic / IDBS.

    The ``policy``/``value`` pair aliases the label so that burke_bf's
    duck-typed helpers (render_svg, validate_solution) work unmodified.
    """

    instance_name: str
    label: str
    height: int
    placements: tuple[Placement, ...]
    skyline: tuple[int, ...]

    @property
    def policy(self) -> "Solution":
        return self

    @property
    def value(self) -> str:
        return self.label
