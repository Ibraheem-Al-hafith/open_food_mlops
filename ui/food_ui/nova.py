"""NOVA group metadata and HTML rendering of prediction results."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Final, Mapping, Optional


@dataclass(frozen=True, slots=True)
class NovaInfo:
    """Display metadata of one NOVA group."""

    group: int
    label: str
    description: str
    color: str
    text_color: str = "#ffffff"


NOVA_INFO: Final[Mapping[int, NovaInfo]] = {
    1: NovaInfo(1, "Unprocessed or minimally processed",
                "Natural foods such as fruit, vegetables, eggs, milk or plain grains.", "#2f8f4e"),
    2: NovaInfo(2, "Processed culinary ingredient",
                "Substances extracted from group 1 foods, such as oils, butter, sugar or salt.",
                "#e3b023", "#2b2100"),
    3: NovaInfo(3, "Processed food",
                "Group 1 foods combined with group 2 ingredients, like bread, cheese or canned vegetables.",
                "#e8741a"),
    4: NovaInfo(4, "Ultra-processed food",
                "Industrial formulations with additives and ingredients rarely used in home cooking.",
                "#c8372d"),
}


def nova_label(group: int) -> str:
    """Short label of a NOVA group; falls back to ``Unknown`` for invalid input."""
    info = NOVA_INFO.get(group)
    return info.label if info else "Unknown"


def render_prediction(group: int, probability: float) -> str:
    """HTML card showing the predicted NOVA group and its confidence."""
    info = NOVA_INFO[group]
    percent = max(0.0, min(1.0, probability)) * 100
    return (
        f'<div class="nova-card" style="--nova:{info.color};--nova-text:{info.text_color}">'
        f'<div class="nova-badge" aria-hidden="true">{info.group}</div>'
        '<div class="nova-body">'
        f'<div class="nova-title">NOVA {info.group}: {html.escape(info.label)}</div>'
        f'<p class="nova-desc">{html.escape(info.description)}</p>'
        f'<div class="nova-conf-label">Model confidence {percent:.0f}%</div>'
        f'<div class="nova-track"><div class="nova-fill" style="width:{percent:.1f}%"></div></div>'
        "</div></div>"
    )


def render_placeholder(message: str) -> str:
    """Neutral panel shown before a result exists."""
    return f'<div class="nova-empty">{html.escape(message)}</div>'


def render_warnings(warnings: list[str]) -> str:
    """HTML list of input plausibility warnings (empty string when none)."""
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return (
        '<div class="nova-warn"><strong>Please double-check these values</strong>'
        f"<ul>{items}</ul></div>"
    )


def render_distribution(counts: Mapping[int, int]) -> str:
    """Horizontal bars with the number of products per NOVA group."""
    total = sum(counts.values())
    rows: list[str] = []
    for group, info in NOVA_INFO.items():
        count = counts.get(group, 0)
        width = (count / total * 100) if total else 0.0
        rows.append(
            '<div class="dist-row">'
            f'<div class="dist-name">NOVA {group}</div>'
            f'<div class="dist-track"><div class="dist-fill" '
            f'style="width:{width:.1f}%;background:{info.color}"></div></div>'
            f'<div class="dist-count">{count:,}</div></div>'
        )
    return f'<div class="dist">{"".join(rows)}</div>'


def render_status(healthy: bool, model_version: Optional[str]) -> str:
    """Service status pill shown in the page header."""
    if healthy:
        version = html.escape(model_version or "unknown")
        return f'<span class="pill pill-ok">Model ready ({version})</span>'
    return '<span class="pill pill-bad">Model unavailable</span>'


def render_batch_summary(
    total: int, predicted: int, rejected: int, model_version: str, counts: Mapping[int, int]
) -> str:
    """Headline numbers plus the NOVA distribution of a finished batch run."""
    rejected_class = " is-warn" if rejected else ""
    return (
        '<div class="summary">'
        f'<div class="stat"><span class="stat-value">{total:,}</span><span class="stat-name">rows uploaded</span></div>'
        f'<div class="stat"><span class="stat-value">{predicted:,}</span><span class="stat-name">predicted</span></div>'
        f'<div class="stat{rejected_class}"><span class="stat-value">{rejected:,}</span>'
        '<span class="stat-name">rejected</span></div>'
        f'<div class="stat"><span class="stat-value stat-small">{html.escape(model_version)}</span>'
        '<span class="stat-name">model version</span></div></div>'
        f"{render_distribution(counts)}"
    )
