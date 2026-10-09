"""The configurable skyline engine (BF frame) for component analysis.

One engine, one array skyline: the lowest gap is located each step, a
rectangle is selected by the configured rule and placed by the configured
policy. All reproduced BF-family heuristics are configurations of it:

- Burke BF:      width / widest-fit / LM|TN|SN (+tower-removal)
- ISH:           area-ish / fitness-number / LM
- TWBF:          {width,height,area} / widest-fit / six policies (+tower, +rotation-rule)
- BBF (frame):   width / tre|nre or widest-fit(first-fit) / LM|TN|SN
- TCBF (control): width / max-area / LM
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, sqrt

from burke_bf.model import Instance, Item, Placement

from .model import Config, Ordering, PlacementPolicy, Selection


@dataclass(frozen=True, slots=True)
class _Rectangle:
    item: Item
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Gap:
    x: int
    y: int
    width: int

    @property
    def right(self) -> int:
        return self.x + self.width


def _sort(
    items: list[Item], ordering: Ordering, strip_width: int, rotatable: bool
) -> list[_Rectangle]:
    if rotatable:
        rectangles = [
            _Rectangle(item, max(item.width, item.height), min(item.width, item.height))
            for item in items
        ]
    else:
        rectangles = [_Rectangle(item, item.width, item.height) for item in items]
    keys = {
        Ordering.WIDTH: lambda r: (-r.width, -r.height),
        Ordering.HEIGHT: lambda r: (-r.height, -r.width),
        Ordering.AREA: lambda r: (-r.width * r.height, -r.width),
        Ordering.PERIMETER: lambda r: (-2 * (r.width + r.height), -r.width),
        Ordering.MAXSIDE: lambda r: (-max(r.width, r.height), -r.height),
        Ordering.DIAGONAL: lambda r: (
            -(sqrt(r.width**2 + r.height**2) + r.width + r.height),
            -r.width,
        ),
    }
    rectangles.sort(key=keys[ordering])
    return rectangles


def _twbf_list(items: list[Item], ordering: Ordering, strip_width: int) -> list:
    """TWBF-style ordered config list (both configurations, oversized replaced)."""

    canonical = [
        (item, max(item.width, item.height), min(item.width, item.height))
        for item in items
    ]
    entries = []
    if ordering is Ordering.WIDTH:
        for item, w, h in canonical:
            entries.append((item, w, h))
            entries.append((item, h, w))
        entries.sort(key=lambda e: (-e[1], -e[2]))
        return [
            (e[0], e[2], e[1]) if e[1] > strip_width else e
            for e in entries
        ]
    if ordering is Ordering.HEIGHT:
        canonical.sort(key=lambda t: (-t[2], -t[1]))
    else:
        canonical.sort(key=lambda t: (-t[1] * t[2], -t[1]))
    for item, w, h in canonical:
        entries.append((item, h, w) if w > strip_width else (item, w, h))
        entries.append((item, h, w))
    return entries


def _lowest_gap(skyline: list[int]) -> _Gap:
    y = min(skyline)
    x = skyline.index(y)
    right = x + 1
    while right < len(skyline) and skyline[right] == y:
        right += 1
    return _Gap(x=x, y=y, width=right - x)


def _neighbours(skyline: list[int], gap: _Gap) -> tuple[float, float]:
    left = skyline[gap.x - 1] if gap.x > 0 else inf
    right = skyline[gap.right] if gap.right < len(skyline) else inf
    return left, right


def _raise(skyline: list[int], gap: _Gap) -> None:
    left, right = _neighbours(skyline, gap)
    new_height = min(left, right)
    if new_height == inf:
        raise RuntimeError("no rectangle fits an empty full-width strip")
    if new_height <= gap.y:
        raise RuntimeError("invalid skyline: neighbour does not bound the gap")
    skyline[gap.x : gap.right] = [int(new_height)] * gap.width


@dataclass(frozen=True, slots=True)
class _VerticalNiche:
    """Region above one skyline segment, extending up to the expected height."""

    x: int
    y: int
    width: int  # horizontal extent (the segment's width)
    vheight: int  # vertical extent, up to the expected best height


def _expected_best_height(instance: Instance) -> int:
    """Continuous bound, lifted to the maximum side length when needed (BBF)."""

    bound = ceil(instance.total_area / instance.strip_width)
    max_side = max(max(item.width, item.height) for item in instance.items)
    if max_side > bound and max_side > instance.strip_width:
        return max_side
    return bound


def _vertical_niche(skyline: list[int], ebh: int) -> _VerticalNiche | None:
    """Region above the leftmost segment not yet reaching the expected height."""

    x = 0
    while x < len(skyline):
        right = x + 1
        while right < len(skyline) and skyline[right] == skyline[x]:
            right += 1
        if skyline[x] < ebh:
            return _VerticalNiche(
                x=x, y=skyline[x], width=right - x, vheight=ebh - skyline[x]
            )
        x = right
    return None


def _exact_vertical(
    remaining: list[_Rectangle], niche: _VerticalNiche
) -> tuple[int, _Rectangle, int, int] | None:
    """Transposed exact fit spanning the niche up to the expected height (eVH)."""

    for index, r in enumerate(remaining):
        if r.width == niche.vheight and r.height <= niche.width:
            return index, r, r.height, r.width
    return None


def _best_vertical(
    remaining: list[_Rectangle], niche: _VerticalNiche
) -> tuple[int, _Rectangle, int, int] | None:
    """First rectangle in list order that fits the niche transposed (WR)."""

    for index, r in enumerate(remaining):
        if r.width <= niche.vheight and r.height <= niche.width:
            return index, r, r.height, r.width
    return None


def _place_on_left(
    policy: PlacementPolicy, skyline: list[int], gap: _Gap, placed_top: int
) -> bool:
    if policy is PlacementPolicy.LM:
        return True
    if policy is PlacementPolicy.RM:
        return False
    left, right = _neighbours(skyline, gap)
    if policy is PlacementPolicy.TN:
        return left >= right
    if policy is PlacementPolicy.MAX_DIFF:
        return abs(left - placed_top) >= abs(right - placed_top)
    # SN and MIN_DIFF never treat the infinitely tall sheet side as the
    # shortest neighbour; bounded on both sides by sheet sides, the rectangle
    # goes to the right (Verstichel's clarification).
    if left == right == inf:
        return False
    if policy is PlacementPolicy.SN:
        return left <= right
    return abs(left - placed_top) <= abs(right - placed_top)


def _first_fit(
    remaining: list[_Rectangle], gap: _Gap, rotatable: bool = True
) -> tuple[int, _Rectangle, int, int] | None:
    """First rectangle in list order that fits the gap.

    The rotated fit is only legal when the configuration is rotatable;
    previously it fired unconditionally even with ``rotatable=False``.
    """

    for index, r in enumerate(remaining):
        if r.width <= gap.width:
            return index, r, r.width, r.height
        if rotatable and r.height <= gap.width:
            return index, r, r.height, r.width
    return None


def _select(
    config: Config,
    remaining: list[_Rectangle],
    skyline: list[int],
    gap: _Gap,
    *,
    fallback: bool = True,
) -> tuple[int, _Rectangle, int, int] | None:
    """Return (index, rectangle, placed_width, placed_height) or None.

    With ``fallback=False`` the niche rules (TRE/NRE) stop after their exact
    scans; the vertical-niche pipeline uses that to interleave the exact
    vertical try before the first-fit fallback (BBF's stage order).
    """

    rule = config.selection
    if config.rotation_rule:
        # Entries are pre-oriented configurations (TWBF-style list); do not
        # re-rotate: an entry fits iff its listed width fits the gap.
        if rule is Selection.WIDEST_FIT:
            best = None
            best_fill = 0
            for index, r in enumerate(remaining):
                # The early break is only sound for width-sorted lists.
                if r.width < best_fill and config.ordering is Ordering.WIDTH:
                    break
                if r.width <= gap.width:
                    if r.width == gap.width:
                        return index, r, r.width, r.height
                    if r.width > best_fill:
                        best = (index, r, r.width, r.height)
                        best_fill = r.width
            return best
        if rule is Selection.FIRST_FIT:
            for index, r in enumerate(remaining):
                if r.width <= gap.width:
                    return index, r, r.width, r.height
            return None
        if rule is Selection.TRE:
            for index, r in enumerate(remaining):
                if r.width == gap.width:
                    return index, r, r.width, r.height
            # First-fit fallback, as in the plain branch: a flat full-width
            # gap cannot be raised.
            for index, r in enumerate(remaining):
                if r.width <= gap.width:
                    return index, r, r.width, r.height
            return None
        raise AssertionError(f"rule {rule} unsupported with rotation_rule")

    if rule is Selection.WIDEST_FIT:
        best = None
        best_fill = 0
        for index, r in enumerate(remaining):
            # The early break is only sound for width-sorted lists.
            if r.width < best_fill and config.ordering is Ordering.WIDTH:
                break
            if r.width <= gap.width:
                w, h = r.width, r.height
            elif config.rotatable and r.height <= gap.width:
                w, h = r.height, r.width
            else:
                continue
            if w == gap.width:
                return index, r, w, h
            if w > best_fill:
                best = (index, r, w, h)
                best_fill = w
        return best

    if rule is Selection.FIRST_FIT:
        return _first_fit(remaining, gap, config.rotatable)

    if rule is Selection.TRE or rule is Selection.NRE:
        if rule is Selection.NRE:
            left, right = _neighbours(skyline, gap)
            for neighbour_height, _on_left in (
                (max(left, right), left >= right),
                (min(left, right), left <= right),
            ):
                if neighbour_height == inf:
                    continue
                target = neighbour_height - gap.y
                for index, r in enumerate(remaining):
                    if r.height == target and r.width <= gap.width:
                        return index, r, r.width, r.height
                    if (
                        config.rotatable
                        and r.width == target
                        and r.height <= gap.width
                    ):
                        return index, r, r.height, r.width
        for index, r in enumerate(remaining):
            if r.width == gap.width:
                return index, r, r.width, r.height
            if config.rotatable and r.height == gap.width:
                return index, r, r.height, r.width
        # First-fit fallback: without it the niche-only rules could strand a
        # flat full-width gap, which cannot be raised (both sheet sides).
        if fallback:
            return _first_fit(remaining, gap, config.rotatable)
        return None

    if rule is Selection.FITNESS_NUMBER:
        left, right = _neighbours(skyline, gap)
        lh = left - gap.y if left != inf else inf
        rh = right - gap.y if right != inf else inf
        best = None
        for index, r in enumerate(remaining):
            if r.width > gap.width and r.height > gap.width:
                continue
            orientations = (
                ((r.width, r.height), (r.height, r.width))
                if config.rotatable
                else ((r.width, r.height),)
            )
            for w, h in orientations:
                if w > gap.width:
                    continue
                fitness = (w == gap.width) + (h == lh) + (h == rh)
                if best is None or fitness > best[0]:
                    best = (fitness, index, r, w, h)
        if best is None:
            return None
        _f, index, r, w, h = best
        return index, r, w, h

    if rule is Selection.MAX_AREA:
        first_c = next((r for r in remaining if r.width <= gap.width), None)
        r_prime = sorted(
            (
                r
                for r in remaining
                if config.rotatable and r.height <= gap.width
            ),
            key=lambda r: (-r.height, -r.width),
        )
        first_r = r_prime[0] if r_prime else None
        if first_c is None and first_r is None:
            return None
        if first_c is None:
            index = remaining.index(first_r)
            return index, first_r, first_r.height, first_r.width
        if first_r is None or first_c is first_r:
            index = remaining.index(first_c)
            return index, first_c, first_c.width, first_c.height
        chosen = (
            first_c if first_c.item.area >= first_r.item.area else first_r
        )
        index = remaining.index(chosen)
        if chosen is first_c:
            return index, chosen, chosen.width, chosen.height
        return index, chosen, chosen.height, chosen.width

    raise AssertionError(f"unknown selection rule {rule}")


def _reduce_towers(
    placements: list[Placement], skyline: list[int], policy: PlacementPolicy
) -> tuple[list[Placement], list[int], int]:
    moves = 0
    while True:
        highest = max(placements, key=lambda placement: placement.top)
        if highest.width >= highest.height or highest.height > len(skyline):
            break
        old_height = max(skyline)
        old_skyline = skyline.copy()
        index = placements.index(highest)
        if any(value != highest.top for value in skyline[highest.x : highest.right]):
            raise RuntimeError("highest rectangle is not exposed across its full width")
        skyline[highest.x : highest.right] = [
            value - highest.height for value in skyline[highest.x : highest.right]
        ]
        del placements[index]
        required = highest.height
        while True:
            gap = _lowest_gap(skyline)
            if required <= gap.width:
                on_left = _place_on_left(policy, skyline, gap, gap.y + highest.width)
                x = gap.x if on_left else gap.right - required
                replacement = Placement(
                    item_id=highest.item_id,
                    x=x,
                    y=gap.y,
                    width=highest.height,
                    height=highest.width,
                    original_width=highest.original_width,
                    original_height=highest.original_height,
                )
                skyline[x : x + replacement.width] = [replacement.top] * replacement.width
                break
            _raise(skyline, gap)
        placements.append(replacement)
        if max(skyline) >= old_height:
            placements.pop()
            placements.insert(index, highest)
            skyline = old_skyline
            break
        moves += 1
    return placements, skyline, moves


def solve_config(
    instance: Instance,
    config: Config,
    sequence: list[Item] | None = None,
) -> tuple[list[Placement], list[int], int]:
    """Pack an instance under one component configuration.

    With ``sequence`` the items are packed in the given order (used by the
    search shells); only orientation normalisation applies, the configured
    ordering is bypassed.
    """

    if config.rotation_rule and config.vertical_niche:
        raise ValueError("rotation_rule and vertical_niche are not combinable")
    if sequence is not None:
        if config.rotatable:
            remaining = [
                _Rectangle(item, max(item.width, item.height), min(item.width, item.height))
                for item in sequence
            ]
        else:
            remaining = [_Rectangle(item, item.width, item.height) for item in sequence]
    elif config.rotation_rule:
        remaining_raw = _twbf_list(list(instance.items), config.ordering, instance.strip_width)
        remaining = [_Rectangle(item, w, h) for item, w, h in remaining_raw]
    else:
        remaining = _sort(
            list(instance.items), config.ordering, instance.strip_width,
            config.rotatable,
        )
    skyline = [0] * instance.strip_width
    placements: list[Placement] = []
    ebh = _expected_best_height(instance) if config.vertical_niche else 0

    while remaining:
        gap = _lowest_gap(skyline)
        niche = (
            _vertical_niche(skyline, ebh) if config.vertical_niche else None
        )
        found = None
        in_niche = False
        if niche is not None and config.selection in (
            Selection.TRE,
            Selection.NRE,
        ):
            # BBF's stage order: horizontal exact, vertical exact (eVH),
            # horizontal best, vertical best (WR).
            found = _select(config, remaining, skyline, gap, fallback=False)
            if found is None:
                found = _exact_vertical(remaining, niche)
                in_niche = found is not None
            if found is None:
                found = _first_fit(remaining, gap, config.rotatable)
        if found is None:
            found = _select(config, remaining, skyline, gap)
        if found is None and niche is not None:
            found = _best_vertical(remaining, niche)
            in_niche = found is not None
        if found is None:
            _raise(skyline, gap)
            continue
        index, rectangle, width, height = found
        if in_niche:
            # Niche placements go at the niche's bottom-left (BBF).
            assert niche is not None
            x, y = niche.x, niche.y
        else:
            y = gap.y
            if config.selection is Selection.FITNESS_NUMBER:
                # ISH's corner rule: the corner with the better fitness, ties
                # to the tallest neighbour (then to the left corner).
                left, right = _neighbours(skyline, gap)
                lh = left - gap.y if left != inf else inf
                rh = right - gap.y if right != inf else inf
                fit_left = (width == gap.width) + (height == lh) + (
                    width == gap.width and height == rh
                )
                fit_right = (width == gap.width) + (height == rh) + (
                    width == gap.width and height == lh
                )
                if fit_left != fit_right:
                    on_left = fit_left > fit_right
                else:
                    on_left = lh >= rh
            else:
                on_left = _place_on_left(
                    config.placement, skyline, gap, gap.y + height
                )
            x = gap.x if on_left else gap.right - width
        placements.append(
            Placement(
                item_id=rectangle.item.item_id,
                x=x,
                y=y,
                width=width,
                height=height,
                original_width=rectangle.item.width,
                original_height=rectangle.item.height,
            )
        )
        skyline[x : x + width] = [y + height] * width
        if config.rotation_rule:
            # TWBF-style lists hold both configurations of each rectangle.
            item_id = remaining[index].item.item_id
            remaining = [r for r in remaining if r.item.item_id != item_id]
        else:
            del remaining[index]

    moves = 0
    if config.tower_removal:
        placements, skyline, moves = _reduce_towers(
            placements, skyline, config.placement
        )
    return placements, skyline, moves
