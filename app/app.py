"""Gradio entry point for the Fly Connectome Explorer.

Assembles the "Graph Theory" and "Neuron Embeddings" tabs (defined in
``graph_tab.py`` and ``embeddings_tab.py``) into a single ``gr.Blocks`` app.
Only reads precomputed artifacts from ``/data`` - no heavy compute happens
at startup or per-request; run the ``/ingest`` pipeline first to produce
them:

.. code-block:: bash

    python ingest/fetch_connectome.py
    python ingest/build_graph.py
    python ingest/compute_graph_stats.py
    python ingest/compute_embeddings.py
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embeddings_tab
import graph_tab

REQUIRED_ARTIFACTS = [
    "graph.gpickle",
    "graph_stats.parquet",
    "graph_summary.json",
    "embeddings.parquet",
]

ABOUT_TEXT = """
### About & Ethics

This app performs **structural graph analysis only** on the FlyWire/BANC
*Drosophila* connectome. Node2vec embeddings and graph statistics describe
wiring topology - they are **not** a model of behavior, cognition,
comprehension, or any cognitive process, and no claims of that kind should
be drawn from this tool. It's an exploratory and educational aid for
looking at connectome structure, not a functional or behavioral model of
the fly brain.

Data: [BANC / FlyWire connectome](https://flywire.ai/), CC-BY licensed. See
the [flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
repository and its associated paper for methodology and provenance.
"""


def _check_artifacts() -> None:
    """Verify the required cached artifacts exist before building the UI.

    :raises FileNotFoundError: if any artifact from :data:`REQUIRED_ARTIFACTS`
        is missing, with a message pointing at the ``/ingest`` pipeline.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    missing = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing cached artifact(s) in {data_dir}: {missing}. "
            "Run the /ingest pipeline first: fetch_connectome.py, build_graph.py, "
            "compute_graph_stats.py, compute_embeddings.py (in that order)."
        )


def build_app() -> gr.Blocks:
    """Assemble the full Gradio app from the two analysis tabs.

    :returns: the constructed ``gr.Blocks`` app, not yet launched.
    """
    with gr.Blocks(title="Fly Connectome Explorer") as demo:
        gr.Markdown("# Fly Connectome Explorer")
        gr.Markdown(
            "Structural graph analysis of the BANC/FlyWire *Drosophila* connectome: "
            "classical graph theory alongside connectivity-based neuron embeddings."
        )
        graph_tab.build()
        embeddings_tab.build()
        with gr.Accordion("About & Ethics", open=False):
            gr.Markdown(ABOUT_TEXT)
    return demo


if __name__ == "__main__":
    _check_artifacts()
    app = build_app()
    app.launch()
