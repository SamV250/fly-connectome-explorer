"""Gradio entry point for the Fly Connectome Explorer.

Assembles the "Graph Theory" and "Neuron Embeddings" tabs (defined in
``graph_tab.py`` and ``embeddings_tab.py``) into a single ``gr.Blocks`` app.
Only reads precomputed artifacts from ``/data`` — no heavy compute happens
at startup or per-request; run the ``/ingest`` pipeline first to produce them.

Implemented in a later commit.
"""

if __name__ == "__main__":
    raise NotImplementedError("app.py: implemented in step 5")
