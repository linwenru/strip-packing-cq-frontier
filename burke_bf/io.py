"""Input parsing for the 2DPackLib ``.ins2D`` format."""

from __future__ import annotations

from pathlib import Path

from .model import Instance, Item


def _instance_name(path: Path) -> str:
    return path.stem.replace("-", "").upper()


def load_instance(path: str | Path) -> Instance:
    """Load and validate one 2DPackLib instance.

    Demands are expanded into physical items. The C7 data all have demand one,
    but expanding here prevents the solver from silently mishandling other
    conforming files.
    """

    source = Path(path)
    rows = [
        line.split()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < 2:
        raise ValueError(f"{source}: expected at least two header lines")

    try:
        item_type_count = int(rows[0][0])
        strip_width, raw_reference_height = map(int, rows[1][:2])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{source}: malformed header") from exc

    if item_type_count <= 0 or strip_width <= 0:
        raise ValueError(f"{source}: counts and strip width must be positive")
    if len(rows) != item_type_count + 2:
        raise ValueError(
            f"{source}: header declares {item_type_count} item types, "
            f"but {len(rows) - 2} rows follow"
        )

    items: list[Item] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows[2:], start=3):
        if len(row) < 6:
            raise ValueError(
                f"{source}:{line_number}: expected six integer columns"
            )
        try:
            raw_id, width, height, demand, maximum_copies, _profit = map(
                int, row[:6]
            )
        except ValueError as exc:
            raise ValueError(
                f"{source}:{line_number}: non-integer value"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{source}:{line_number}: item dimensions must be positive"
            )
        if demand <= 0 or maximum_copies < demand:
            raise ValueError(
                f"{source}:{line_number}: invalid demand/copy bounds"
            )
        if min(width, height) > strip_width:
            raise ValueError(
                f"{source}:{line_number}: item {raw_id} cannot fit the strip "
                "even after rotation"
            )

        for copy_index in range(1, demand + 1):
            item_id = str(raw_id) if demand == 1 else f"{raw_id}.{copy_index}"
            if item_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate item ID")
            seen_ids.add(item_id)
            items.append(Item(item_id=item_id, width=width, height=height))

    return Instance(
        name=_instance_name(source),
        strip_width=strip_width,
        reference_height=(
            raw_reference_height if raw_reference_height >= 0 else None
        ),
        items=tuple(items),
    )
