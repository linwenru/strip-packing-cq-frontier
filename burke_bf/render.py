"""Dependency-free SVG rendering for packing inspection."""

from __future__ import annotations

import colorsys
from html import escape
from math import floor, log10

from .model import Instance, Solution


def _colour(item_id: str) -> str:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(item_id))
    red, green, blue = colorsys.hsv_to_rgb((seed * 0.61803398875) % 1, 0.42, 0.92)
    return f"rgb({round(red * 255)},{round(green * 255)},{round(blue * 255)})"


def _tick_step(maximum: int, target_ticks: int = 10) -> int:
    """Choose a readable 1/2/2.5/5/10-style integer tick interval."""

    if maximum <= target_ticks:
        return 1
    rough_step = maximum / target_ticks
    magnitude = 10 ** floor(log10(rough_step))
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if step >= rough_step:
            return max(1, round(step))
    raise AssertionError("unreachable")


def _ticks(maximum: int) -> list[int]:
    step = _tick_step(maximum)
    ticks = list(range(0, maximum + 1, step))
    if ticks[-1] != maximum:
        ticks.append(maximum)
    return ticks


def render_svg(
    instance: Instance,
    solution: Solution,
    *,
    scale: int = 3,
) -> str:
    """Render a complete packing as a standalone SVG string."""

    left_margin = 50
    right_margin = 20
    top_margin = 38
    bottom_margin = 48
    plot_width = instance.strip_width * scale
    plot_height = solution.height * scale
    canvas_width = plot_width + left_margin + right_margin
    canvas_height = plot_height + top_margin + bottom_margin
    plot_x = left_margin
    plot_y = top_margin
    axis_bottom = plot_y + plot_height
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_width}" height="{canvas_height}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{plot_x}" y="21" font-family="sans-serif" '
            f'font-size="13">{escape(instance.name)} - '
            f'{escape(solution.policy.value)} - H={solution.height}</text>'
        ),
        (
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" '
            f'height="{plot_height}" fill="#f6f6f6"/>'
        ),
        '<g id="grid" stroke="#d8d8d8" stroke-width="0.6">',
    ]

    x_ticks = _ticks(instance.strip_width)
    y_ticks = _ticks(solution.height)
    sequence_by_id = {
        item.item_id: sequence
        for sequence, item in enumerate(instance.items, start=1)
    }
    for value in x_ticks:
        x = plot_x + value * scale
        parts.append(
            f'<line x1="{x}" y1="{plot_y}" x2="{x}" y2="{axis_bottom}"/>'
        )
    for value in y_ticks:
        y = axis_bottom - value * scale
        parts.append(
            f'<line x1="{plot_x}" y1="{y}" '
            f'x2="{plot_x + plot_width}" y2="{y}"/>'
        )
    parts.append("</g>")

    for placement in solution.placements:
        x = plot_x + placement.x * scale
        y = plot_y + (solution.height - placement.top) * scale
        rect_width = placement.width * scale
        rect_height = placement.height * scale
        parts.append(
            f'<rect x="{x}" y="{y}" width="{rect_width}" height="{rect_height}" '
            f'fill="{_colour(placement.item_id)}" stroke="#222" '
            f'stroke-width="0.6"><title>item {escape(placement.item_id)}: '
            f'({placement.x},{placement.y}) '
            f'{placement.width}x{placement.height}</title></rect>'
        )
        label_size = max(2, min(10, rect_width * 0.55, rect_height * 0.65))
        parts.append(
            f'<text class="item-label" x="{x + rect_width / 2}" '
            f'y="{y + rect_height / 2}" text-anchor="middle" '
            f'dominant-baseline="central" font-family="sans-serif" '
            f'font-size="{label_size:.2f}" fill="#111" stroke="white" '
            f'stroke-width="0.7" paint-order="stroke" '
            f'pointer-events="none">{sequence_by_id[placement.item_id]}</text>'
        )

    parts.extend(
        [
            (
                f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" '
                f'height="{plot_height}" fill="none" stroke="#111" '
                f'stroke-width="1"/>'
            ),
            '<g id="x-axis" font-family="sans-serif" font-size="10" fill="#222">',
        ]
    )
    for value in x_ticks:
        x = plot_x + value * scale
        parts.append(
            f'<line x1="{x}" y1="{axis_bottom}" x2="{x}" '
            f'y2="{axis_bottom + 5}" stroke="#111"/>'
        )
        parts.append(
            f'<text x="{x}" y="{axis_bottom + 17}" '
            f'text-anchor="middle">{value}</text>'
        )
    parts.append(
        f'<text x="{plot_x + plot_width / 2}" y="{canvas_height - 7}" '
        f'text-anchor="middle" font-size="12">x</text>'
    )
    parts.extend(
        [
            "</g>",
            '<g id="y-axis" font-family="sans-serif" font-size="10" fill="#222">',
        ]
    )
    for value in y_ticks:
        y = axis_bottom - value * scale
        parts.append(
            f'<line x1="{plot_x - 5}" y1="{y}" x2="{plot_x}" '
            f'y2="{y}" stroke="#111"/>'
        )
        parts.append(
            f'<text x="{plot_x - 8}" y="{y + 3.5}" '
            f'text-anchor="end">{value}</text>'
        )
    parts.append(
        f'<text x="13" y="{plot_y + plot_height / 2}" font-size="12" '
        f'text-anchor="middle" '
        f'transform="rotate(-90 13 {plot_y + plot_height / 2})">y</text>'
    )
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"
