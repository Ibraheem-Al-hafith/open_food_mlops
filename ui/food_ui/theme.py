"""Visual identity of the UI: a restrained theme plus the stylesheet for custom HTML."""

from __future__ import annotations

from typing import Final

import gradio as gr

TITLE: Final[str] = "Open Food NOVA classifier"

HEADER_HTML: Final[str] = (
    '<div class="app-header"><h1>Open Food NOVA classifier</h1>'
    "<p>Estimate how processed a food is from its nutrition label (values per 100 g).</p></div>"
)


def build_theme() -> gr.themes.Base:
    """System fonts only, so the page makes no third-party requests."""
    return gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=("ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"),
    )


CSS: Final[str] = """
.app-header { border-top: 8px solid var(--body-text-color); padding: 12px 0 8px; }
.app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em; }
.app-header p { margin: 4px 0 0; color: var(--body-text-color-subdued); }

.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.pill-ok { background: #dcf3e4; color: #14532d; }
.pill-bad { background: #fde1de; color: #7f1d1d; }

.nova-card { display: flex; border: 1px solid var(--border-color-primary); border-radius: 8px;
  overflow: hidden; background: var(--background-fill-secondary); }
.nova-badge { flex: 0 0 6rem; display: grid; place-items: center; font-size: 4.5rem; font-weight: 800;
  background: var(--nova); color: var(--nova-text); }
.nova-body { padding: 16px 18px; flex: 1; }
.nova-title { font-size: 1.15rem; font-weight: 700; }
.nova-desc { margin: 6px 0 14px; color: var(--body-text-color-subdued); max-width: 60ch; }
.nova-conf-label { font-size: 0.9rem; margin-bottom: 4px; }
.nova-track { height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; }
.nova-fill { height: 100%; background: var(--nova); }
.nova-empty { padding: 28px 18px; border: 1px dashed var(--border-color-primary); border-radius: 8px;
  color: var(--body-text-color-subdued); }
.nova-warn { margin-top: 12px; padding: 12px 16px; border-left: 4px solid #e8741a;
  background: var(--background-fill-secondary); border-radius: 4px; }
.nova-warn ul { margin: 6px 0 0 18px; padding: 0; }

.summary { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
.stat { flex: 1 1 140px; padding: 12px 14px; border: 1px solid var(--border-color-primary);
  border-radius: 8px; background: var(--background-fill-secondary); display: flex; flex-direction: column; }
.stat.is-warn { border-color: #e8741a; }
.stat-value { font-size: 1.6rem; font-weight: 800; }
.stat-small { font-size: 1rem; padding-top: 0.5rem; overflow-wrap: anywhere; }
.stat-name { color: var(--body-text-color-subdued); font-size: 0.85rem; }

.dist { display: grid; gap: 6px; margin-bottom: 8px; }
.dist-row { display: grid; grid-template-columns: 5rem 1fr 4rem; gap: 10px; align-items: center; }
.dist-track { height: 14px; background: var(--border-color-primary); border-radius: 3px; overflow: hidden; }
.dist-fill { height: 100%; }
.dist-count { text-align: right; font-variant-numeric: tabular-nums; }

@media (max-width: 640px) {
  .nova-badge { flex-basis: 4rem; font-size: 3rem; }
}
"""
