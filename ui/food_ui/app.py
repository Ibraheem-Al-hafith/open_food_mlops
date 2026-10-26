"""Gradio application: layout and entry point (``python -m food_ui.app``)."""

from __future__ import annotations

import logging

import gradio as gr

from food_ui.batch import SUPPORTED_SUFFIXES
from food_ui.client import PredictionClient
from food_ui.config import UiSettings
from food_ui.controller import BATCH_PLACEHOLDER, RESULT_PLACEHOLDER, UiController
from food_ui.features import FEATURE_GROUPS, FEATURES, component_label
from food_ui.nova import render_placeholder
from food_ui.theme import CSS, HEADER_HTML, TITLE, build_theme

logger = logging.getLogger(__name__)


def build_demo(controller: UiController, max_batch_rows: int) -> gr.Blocks:
    """Create the two-tab interface and wire it to ``controller``."""
    demo: gr.Blocks
    with gr.Blocks(title=TITLE) as demo:
        with gr.Row(equal_height=True):
            gr.HTML(HEADER_HTML)
            status = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Single product"):
                _build_online_tab(controller)
            with gr.Tab("Batch"):
                _build_batch_tab(controller, max_batch_rows)

        demo.load(controller.status_html, outputs=status)

    return demo


def _build_online_tab(controller: UiController) -> None:
    with gr.Row():
        with gr.Column(scale=3):
            example = gr.Dropdown(
                choices=controller.example_names(),
                value=None,
                label="Start from an example",
                info="Approximate values, meant for trying the form.",
            )
            product_code = gr.Textbox(
                label="Product code (optional)",
                info="Barcode or your own identifier; stored with the prediction.",
            )

            inputs: list[gr.Number] = []
            for group in FEATURE_GROUPS:
                with gr.Accordion(group, open=True):
                    with gr.Row():
                        for spec in (s for s in FEATURES if s.group == group):
                            inputs.append(
                                gr.Number(
                                    label=component_label(spec),
                                    value=0,
                                    minimum=0,
                                    maximum=spec.maximum,
                                    info=spec.help_text or None,
                                    min_width=160,
                                )
                            )

            with gr.Row():
                predict_button = gr.Button("Predict", variant="primary")
                clear_button = gr.Button("Clear")

        with gr.Column(scale=2):
            result = gr.HTML(render_placeholder(RESULT_PLACEHOLDER))
            warnings = gr.HTML()

    example.change(controller.load_example, inputs=example, outputs=[product_code, *inputs])
    clear_button.click(controller.clear, outputs=[product_code, *inputs, result, warnings])
    for trigger in (predict_button.click, product_code.submit):
        trigger(
            controller.predict_online,
            inputs=[product_code, *inputs],
            outputs=[result, warnings],
        )


def _build_batch_tab(controller: UiController, max_batch_rows: int) -> None:
    accepted = ", ".join(SUPPORTED_SUFFIXES)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                f"Upload up to **{max_batch_rows:,} rows** ({accepted}). "
                "Each row needs the 14 nutrition columns; `product_code` is optional."
            )
            gr.DownloadButton("Download template", value=controller.template_path, size="sm")
            upload = gr.File(label="Products file", file_types=list(SUPPORTED_SUFFIXES), type="filepath")
            persist = gr.Checkbox(
                label="Save predictions to the monitoring store",
                value=False,
                info="Leave off for experiments. Saved predictions count towards drift and performance monitoring.",
            )
            run_button = gr.Button("Run batch", variant="primary")

        with gr.Column(scale=3):
            summary = gr.HTML(render_placeholder(BATCH_PLACEHOLDER))
            table = gr.Dataframe(label="Results", interactive=False, max_height=420)
            export = gr.File(label="Download results", interactive=False)

    run_button.click(controller.run_batch, inputs=[upload, persist], outputs=[summary, table, export])


def main() -> None:
    """Launch the UI using ``UI_*`` environment variables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = UiSettings()
    client = PredictionClient(settings.api_url, timeout_s=settings.request_timeout_s)
    controller = UiController(client, settings)

    demo = build_demo(controller, settings.max_batch_rows)
    demo.queue(default_concurrency_limit=4)

    logger.info("Starting UI on %s:%d (api=%s, root_path=%r)",
                settings.host, settings.port, settings.api_url, settings.root_path)
    demo.launch(
        server_name=settings.host,
        server_port=settings.port,
        root_path=settings.root_path or None,
        auth=settings.auth,
        max_file_size="20mb",
        footer_links=["gradio"],
        theme=build_theme(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
