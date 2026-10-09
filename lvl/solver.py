"""Level-packing heuristics: FFDH, BFDH and BFS (best-fit with stacking).

FFDH (Coffman et al. 1980) packs items sorted by decreasing height (ties by
decreasing width) into the lowest level with enough residual width. BFDH
(Coffman and Shor 1990) picks the level with the minimum residual width.
BFS (Ortmann, Ntene and van Vuuren 2010, Algorithm 2) improves on BFDH-style
packing: after each floor placement, the free rectangle above the item is
filled with further items FFDH-style, expanding its right boundary to the
strip edge when the space next to the item cannot hold the narrowest
unpacked item. All packings are oriented (no rotation) and level-based.
"""

from __future__ import annotations

from dataclasses import dataclass

from burke_bf.model import Item, Placement


def sort_dhdw(items: list[Item]) -> list[Item]:
    """Decreasing height, ties by decreasing width (DHDW)."""

    return sorted(items, key=lambda item: (-item.height, -item.width))


def sort_dwdh(items: list[Item]) -> list[Item]:
    """Decreasing width, ties by decreasing height (DWDH)."""

    return sorted(items, key=lambda item: (-item.width, -item.height))


@dataclass(slots=True)
class _Level:
    y: int
    height: int
    used: int  # width already consumed on the floor

    @property
    def ceiling(self) -> int:
        return self.y + self.height


def _pack_into(level: _Level, item: Item, x: int) -> Placement:
    level.used = max(level.used, x + item.width)
    return Placement(
        item_id=item.item_id,
        x=x,
        y=level.y,
        width=item.width,
        height=item.height,
        original_width=item.width,
        original_height=item.height,
    )


def _new_level(levels: list[_Level], item: Item) -> _Level:
    y = levels[-1].ceiling if levels else 0
    level = _Level(y=y, height=item.height, used=0)
    levels.append(level)
    return level


def pack_ffdh(
    items: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """First-fit decreasing height: lowest level with residual width."""

    levels: list[_Level] = []
    placements: list[Placement] = []
    for item in sort_dhdw(items):
        level = next(
            (
                candidate
                for candidate in levels
                if strip_width - candidate.used >= item.width
            ),
            None,
        )
        if level is None:
            level = _new_level(levels, item)
        placements.append(_pack_into(level, item, level.used))
    return placements, levels[-1].ceiling


def pack_bfdh(
    items: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """Best-fit decreasing height: level with minimum residual width."""

    levels: list[_Level] = []
    placements: list[Placement] = []
    for item in sort_dhdw(items):
        fitting = [
            candidate
            for candidate in levels
            if strip_width - candidate.used >= item.width
        ]
        level = (
            min(fitting, key=lambda candidate: candidate.used)
            if fitting
            else None
        )
        if level is None:
            level = _new_level(levels, item)
        placements.append(_pack_into(level, item, level.used))
    return placements, levels[-1].ceiling


def fill_rectangle_ffdh(
    x0: int,
    y0: int,
    width: int,
    height: int,
    remaining: list[Item],
    placements: list[Placement],
) -> None:
    """Fill a free rectangle with items from the sorted list, FFDH-style.

    The rectangle is treated as a mini-strip of the given width and height:
    items are packed level by level from the bottom (each item into the
    lowest row with enough residual width and vertical room), leaving those
    that fit nowhere unpacked.
    """

    rows: list[list[int]] = []  # [y, row_height, used_width]
    for item in list(remaining):
        placed = False
        for row in rows:
            if (
                row[2] + item.width <= width
                and row[0] + item.height <= y0 + height
            ):
                placements.append(
                    Placement(
                        item_id=item.item_id,
                        x=x0 + row[2],
                        y=row[0],
                        width=item.width,
                        height=item.height,
                        original_width=item.width,
                        original_height=item.height,
                    )
                )
                row[2] += item.width
                placed = True
                break
        if not placed:
            row_y = rows[-1][0] + rows[-1][1] if rows else y0
            if row_y + item.height <= y0 + height and item.width <= width:
                rows.append([row_y, item.height, 0])
                placements.append(
                    Placement(
                        item_id=item.item_id,
                        x=x0,
                        y=row_y,
                        width=item.width,
                        height=item.height,
                        original_width=item.width,
                        original_height=item.height,
                    )
                )
                rows[-1][2] += item.width
                placed = True
            # Otherwise the item does not fit into the rectangle at all;
            # it stays unpacked (later items may still fit).
        if placed:
            remaining.remove(item)

    return rows[0][2] if rows else 0


def pack_bfs(
    items: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """Ortmann et al.'s best-fit with stacking (Algorithm 2)."""

    remaining = sort_dhdw(items)
    levels: list[_Level] = []
    placements: list[Placement] = []

    while remaining:
        item = remaining.pop(0)
        fitting = [
            candidate
            for candidate in levels
            if strip_width - candidate.used >= item.width
        ]
        level = (
            min(fitting, key=lambda candidate: candidate.used)
            if fitting
            else None
        )
        if level is None:
            level = _new_level(levels, item)
        x = level.used
        placements.append(_pack_into(level, item, x))

        # Free rectangle above the item: item.width wide, up to the ceiling.
        rect_x = x
        rect_w = item.width
        # Expand it to the strip edge if the space next to the item cannot
        # hold the narrowest unpacked item.
        if remaining and strip_width - level.used < min(
            other.width for other in remaining
        ):
            rect_w = strip_width - rect_x
        rect_h = level.ceiling - (level.y + item.height)
        fill_rectangle_ffdh(
            rect_x, level.y + item.height, rect_w, rect_h, remaining, placements
        )

    return placements, levels[-1].ceiling


def pack_sasm(
    items: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """Ortmann et al.'s modified size-alternating stack (Algorithm 1).

    Built on the original SAS semantics (Ntene and van Vuuren 2009): narrow
    (h > w, DHDW) and wide (w >= h, DWDH) items alternate between two
    procedures — PackWide stacks wide items from the floor upwards, filling
    the wedges created by unequal widths with narrow items; PackNarrow packs
    a narrow floor item and fills the rectangle above it. SASm's changes are
    applied: DHDW/DWDH tie-breaking, level initialisation with the tallest
    wide item versus the first narrow one, filling the free rectangle above
    the last stacked wide item, and side-by-side narrow stacking (all
    rectangle fills are mini-strip FFDH). When the designated list cannot
    place, the alternative list is tried before finishing the level.
    """

    narrow = sort_dhdw([item for item in items if item.height > item.width])
    wide = sort_dwdh([item for item in items if item.width >= item.height])
    levels: list[_Level] = []
    placements: list[Placement] = []
    last: str | None = None  # branch used most recently; None = new level

    while narrow or wide:
        if last is None:
            # SASm: compare the tallest wide item with the first narrow one.
            tallest_wide = (
                max(wide, key=lambda item: item.height) if wide else None
            )
            first_narrow = narrow[0] if narrow else None
            if tallest_wide is not None and (
                first_narrow is None
                or tallest_wide.height >= first_narrow.height
            ):
                initial, last = tallest_wide, "wide"
                wide.remove(initial)
            else:
                initial, last = first_narrow, "narrow"
                narrow.remove(initial)
            y = levels[-1].ceiling if levels else 0
            level = _Level(y=y, height=initial.height, used=0)
            levels.append(level)
            placements.append(_pack_into(level, initial, 0))
            continue

        level = levels[-1]
        prefer = "wide" if last == "narrow" else "narrow"
        placed_any = False
        for branch in (prefer, last):
            if (
                branch == "wide"
                and wide
                and level.used + wide[0].width <= strip_width
                and level.y + wide[0].height <= level.ceiling
            ):
                # PackWide: stack wide items from the floor upwards.
                x = level.used
                stack_top = level.y
                prev_width: int | None = None
                prev_top = 0
                first_width = 0
                while (
                    wide
                    and x + wide[0].width <= strip_width
                    and stack_top + wide[0].height <= level.ceiling
                    and (prev_width is None or wide[0].width <= prev_width)
                ):
                    item = wide.pop(0)
                    placements.append(
                        Placement(
                            item_id=item.item_id,
                            x=x,
                            y=stack_top,
                            width=item.width,
                            height=item.height,
                            original_width=item.width,
                            original_height=item.height,
                        )
                    )
                    if prev_width is not None and item.width < prev_width:
                        # Wedge beside the narrower item, above the wider one.
                        fill_rectangle_ffdh(
                            x + item.width, prev_top, prev_width - item.width,
                            level.ceiling - prev_top, narrow, placements,
                        )
                    prev_width = item.width
                    prev_top = stack_top + item.height
                    stack_top = prev_top
                    if not first_width:
                        first_width = item.width
                level.used = x + first_width
                # SASm (iii): fill the free rectangle above the last wide item.
                if prev_width is not None:
                    fill_rectangle_ffdh(
                        x, stack_top, prev_width, level.ceiling - stack_top,
                        narrow, placements,
                    )
                last = "wide"
                placed_any = True
                break
            if (
                branch == "narrow"
                and narrow
                and level.used + narrow[0].width <= strip_width
                and level.y + narrow[0].height <= level.ceiling
            ):
                # PackNarrow: a narrow floor item, then SASm (iv) side-by-side
                # stacking in the rectangle above it.
                item = narrow.pop(0)
                x = level.used
                placements.append(_pack_into(level, item, x))
                fill_rectangle_ffdh(
                    x, level.y + item.height, item.width,
                    level.ceiling - (level.y + item.height), narrow, placements,
                )
                last = "narrow"
                placed_any = True
                break
        if not placed_any:
            last = None  # neither list fits; finish the level

    return placements, levels[-1].ceiling


def _ceiling_floor_top(
    placements: list[Placement], level: _Level, x0: int, x1: int
) -> int:
    """Highest top of floor items in the level overlapping [x0, x1)."""

    top = level.y
    for placement in placements:
        if placement.y != level.y:
            continue
        if placement.right <= x0 or placement.x >= x1:
            continue
        top = max(top, placement.top)
    return top


def _pack_ceiling(
    level: _Level,
    remaining: list[Item],
    placements: list[Placement],
    strip_width: int,
    *,
    pick_widest: bool,
) -> None:
    """Stack items downward onto the ceiling, right to left.

    The ceiling is mirrored as a downward skyline d (the lowest occupied y
    from above) over the floor skyline (highest tops below). Pockets are the
    maximal runs where d and the floor tops are both constant; their depth is
    d - floor_top. Each step fills the deepest pocket (rightmost on ties)
    with the tallest fitting item (widest when re-sorting), its top touching
    d; d then shrinks by the item's height. This realises the paper's
    "top-right corner, packing downwards, from right to left" recursion,
    including wedges when the item below is wider than the one above.
    """

    floor_tops = [level.y] * strip_width
    for placement in placements:
        if placement.y != level.y:
            continue
        for x in range(placement.x, placement.right):
            floor_tops[x] = max(floor_tops[x], placement.top)
    d = [level.ceiling] * strip_width

    while True:
        # Pockets: maximal runs where d and the floor tops are both constant.
        pockets = []
        x = 0
        while x < strip_width:
            right = x + 1
            while (
                right < strip_width
                and d[right] == d[x]
                and floor_tops[right] == floor_tops[x]
            ):
                right += 1
            depth = d[x] - floor_tops[x]
            if depth > 0:
                pockets.append((depth, x, right))
            x = right
        if not pockets:
            return

        chosen_pos = None
        chosen_item = None
        chosen_key = None
        for depth, x0, x1 in pockets:
            for item in remaining:
                if item.height > depth or item.width > x1 - x0:
                    continue
                if pick_widest:
                    key = (item.width, item.height)
                else:
                    key = (item.height, item.width)
                pos_key = (depth, x1)
                if (
                    chosen_item is None
                    or pos_key > chosen_pos[0]
                    or (pos_key == chosen_pos[0] and key > chosen_key)
                ):
                    chosen_pos = (pos_key, x0, x1)
                    chosen_item = item
                    chosen_key = key
        if chosen_item is None:
            return

        _pos, x0, x1 = chosen_pos
        x = x1 - chosen_item.width
        placements.append(
            Placement(
                item_id=chosen_item.item_id,
                x=x,
                y=d[x] - chosen_item.height,
                width=chosen_item.width,
                height=chosen_item.height,
                original_width=chosen_item.width,
                original_height=chosen_item.height,
            )
        )
        remaining.remove(chosen_item)
        for xx in range(x, x + chosen_item.width):
            d[xx] -= chosen_item.height


def pack_sc(
    items: list[Item], strip_width: int, *, resort: bool = False
) -> tuple[list[Placement], int]:
    """Ortmann et al.'s stack ceiling algorithm, with optional re-sorting (SCR)."""

    remaining = sort_dhdw(items)
    levels: list[_Level] = []
    placements: list[Placement] = []

    while remaining:
        y = levels[-1].ceiling if levels else 0
        level = _Level(y=y, height=remaining[0].height, used=0)
        levels.append(level)
        # Fill the floor FFDH-style within this level.
        for item in list(remaining):
            if (
                item.width <= strip_width - level.used
                and item.height <= level.height
            ):
                placements.append(_pack_into(level, item, level.used))
                remaining.remove(item)
        if resort:
            remaining = sort_dwdh(remaining)
        # Stack items downward onto the ceiling from the top-right corner.
        _pack_ceiling(
            level, remaining, placements, strip_width, pick_widest=resort
        )
        if resort:
            remaining = sort_dhdw(remaining)

    return placements, levels[-1].ceiling


def pack_sas(
    items: list[Item], strip_width: int
) -> tuple[list[Placement], int]:
    """The original size-alternating stack (Ntene and van Vuuren 2009).

    As SASm but with the original SAS rules: level initialisation comparing
    the first narrow (tallest) versus the first wide (widest), and narrow
    stacking constrained to the width of the bottom-most narrow rectangle.
    """

    narrow = sorted(
        [item for item in items if item.height > item.width],
        key=lambda item: -item.height,
    )
    wide = sorted(
        [item for item in items if item.width >= item.height],
        key=lambda item: -item.width,
    )
    levels: list[_Level] = []
    placements: list[Placement] = []
    last: str | None = None

    def stack_narrow(x: int, y0: int, width: int, height: int) -> int:
        """SAS PackNarrow: floor narrow + stack of narrower ones on top."""

        if not narrow or narrow[0].width > width:
            return 0
        bottom = narrow.pop(0)
        if y0 + bottom.height > height:
            narrow.insert(0, bottom)
            return 0
        placements.append(
            Placement(
                item_id=bottom.item_id, x=x, y=y0, width=bottom.width,
                height=bottom.height, original_width=bottom.width,
                original_height=bottom.height,
            )
        )
        top = y0 + bottom.height
        while narrow and narrow[0].width <= min(bottom.width, width - 0) and top + narrow[0].height <= height:
            item = narrow.pop(0)
            placements.append(
                Placement(
                    item_id=item.item_id, x=x, y=top, width=item.width,
                    height=item.height, original_width=item.width,
                    original_height=item.height,
                )
            )
            top += item.height
        return bottom.width

    while narrow or wide:
        if last is None:
            fn = narrow[0] if narrow else None
            fw = wide[0] if wide else None
            if fw is not None and (fn is None or fw.height >= fn.height):
                initial, last = fw, "wide"
                wide.pop(0)
            else:
                initial, last = fn, "narrow"
                narrow.pop(0)
            y = levels[-1].ceiling if levels else 0
            level = _Level(y=y, height=initial.height, used=0)
            levels.append(level)
            placements.append(_pack_into(level, initial, 0))
            continue

        level = levels[-1]
        prefer = "wide" if last == "narrow" else "narrow"
        placed_any = False
        for branch in (prefer, last):
            if (
                branch == "wide"
                and wide
                and level.used + wide[0].width <= strip_width
                and level.y + wide[0].height <= level.ceiling
            ):
                x = level.used
                stack_top = level.y
                prev_width: int | None = None
                prev_top = 0
                first_width = 0
                while (
                    wide
                    and x + wide[0].width <= strip_width
                    and stack_top + wide[0].height <= level.ceiling
                ):
                    item = wide.pop(0)
                    placements.append(
                        Placement(
                            item_id=item.item_id, x=x, y=stack_top,
                            width=item.width, height=item.height,
                            original_width=item.width,
                            original_height=item.height,
                        )
                    )
                    if prev_width is not None and item.width < prev_width:
                        stack_narrow(
                            x + item.width, prev_top,
                            prev_width - item.width, level.ceiling,
                        )
                    prev_width = item.width
                    prev_top = stack_top + item.height
                    stack_top = prev_top
                    if not first_width:
                        first_width = item.width
                level.used = x + first_width
                last = "wide"
                placed_any = True
                break
            if (
                branch == "narrow"
                and narrow
                and narrow[0].width <= strip_width - level.used
                and level.y + narrow[0].height <= level.ceiling
            ):
                x = level.used
                level.used = x + stack_narrow(
                    x, level.y, strip_width - x, level.ceiling
                )
                last = "narrow"
                placed_any = True
                break
        if not placed_any:
            last = None

    return placements, levels[-1].ceiling
