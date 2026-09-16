"""Neuron Embeddings tab for the Gradio app.

Renders a 2D projection of node2vec embeddings colored by known cell
type/region, a neuron-ID search box that highlights nearest neighbors in
embedding space, and a caption relating structural to functional
similarity. Sourced from ``data/embeddings.parquet`` and
``data/graph_stats.parquet`` (for the Louvain community column). Loads its
cached artifacts once at import time; the only per-request work is a
nearest-neighbor lookup among 1,000 precomputed 64-d vectors, which is
milliseconds of linear algebra, not graph or embedding computation.
"""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLOR_BY_CHOICES = ["super_class", "cell_class", "embedding_cluster", "primary_neuropil"]
PROJECTION_CHOICES = ["pca", "umap"]
NUM_NEIGHBORS = 10

_embeddings_df = pd.read_parquet(DATA_DIR / "embeddings.parquet")
_community = pd.read_parquet(DATA_DIR / "graph_stats.parquet")["community"]
_embeddings_df = _embeddings_df.join(_community)

_vectors = np.stack(_embeddings_df["embedding"].to_numpy())
_node_ids = _embeddings_df.index.to_numpy()
_node_id_to_row = {node_id: i for i, node_id in enumerate(_node_ids)}


def _agreement_caption() -> str:
    """Summarize how well embedding clusters track known/structural labels.

    Computed once at import time (over 1,000 elements - negligible cost),
    not recomputed per request.

    :returns: a Markdown caption comparing node2vec clusters against
        ``super_class`` cell-type labels and against the Graph Theory tab's
        Louvain communities, via Adjusted Rand Index (0 = random, 1 = exact
        agreement).
    """
    valid = _embeddings_df["super_class"].notna()
    ari_cell_type = adjusted_rand_score(
        _embeddings_df.loc[valid, "super_class"], _embeddings_df.loc[valid, "embedding_cluster"]
    )
    ari_community = adjusted_rand_score(
        _embeddings_df["community"], _embeddings_df["embedding_cluster"]
    )
    return (
        f"**Are structurally similar neurons functionally similar?** Clustering the node2vec "
        f"embeddings (which see only wiring, never cell-type labels) reaches an Adjusted Rand "
        f"Index of **{ari_cell_type:.2f}** against known `super_class` annotations, and "
        f"**{ari_community:.2f}** against the Graph Theory tab's Louvain communities from the "
        f"raw topology (1.0 = exact agreement, 0.0 = no better than random). Partial, "
        f"well-above-zero agreement is expected: connectivity correlates with cell identity, "
        f"but wiring alone is not a complete proxy for it."
    )


def build_projection_figure(color_by: str = "super_class", projection: str = "pca", highlight_id: str = "") -> go.Figure:
    """Build the 2D embedding scatter plot, optionally highlighting a neuron.

    :param color_by: column to color points by (one of :data:`COLOR_BY_CHOICES`).
    :param projection: which precomputed 2D projection to use, ``"pca"`` or ``"umap"``.
    :param highlight_id: if a valid neuron root ID (as a string), that neuron
        and its :data:`NUM_NEIGHBORS` nearest neighbors in embedding space
        are drawn with enlarged, outlined markers.
    :returns: a Plotly scatter figure.
    """
    x_col, y_col = (f"{projection}_x", f"{projection}_y")
    color_values = _embeddings_df[color_by].fillna("unknown").astype(str)

    fig = px.scatter(
        _embeddings_df,
        x=x_col,
        y=y_col,
        color=color_values,
        hover_name=_embeddings_df.index.astype(str),
        hover_data={"cell_type": True, "primary_neuropil": True, x_col: False, y_col: False},
        labels={"color": color_by},
        title=f"Node2vec embeddings ({projection.upper()} projection), colored by {color_by}",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))

    highlight_ids = _resolve_highlight_ids(highlight_id)
    if highlight_ids:
        highlight_rows = _embeddings_df.loc[highlight_ids]
        fig.add_trace(
            go.Scatter(
                x=highlight_rows[x_col],
                y=highlight_rows[y_col],
                mode="markers",
                marker=dict(
                    size=[16] + [11] * (len(highlight_ids) - 1),
                    color="rgba(0,0,0,0)",
                    line=dict(width=3, color="black"),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="highlighted",
            )
        )

    fig.update_layout(margin=dict(t=40))
    return fig


def _resolve_highlight_ids(highlight_id: str) -> list:
    """Parse a search box value into the queried node ID plus its neighbors.

    :param highlight_id: raw text from the search box.
    :returns: list of node IDs to highlight (queried node first), or an
        empty list if ``highlight_id`` is blank or not a known neuron.
    """
    highlight_id = highlight_id.strip()
    if not highlight_id:
        return []
    try:
        node_id = int(highlight_id)
    except ValueError:
        return []
    if node_id not in _node_id_to_row:
        return []
    neighbor_ids, _ = nearest_neighbors(node_id, k=NUM_NEIGHBORS)
    return [node_id] + neighbor_ids


def nearest_neighbors(node_id: int, k: int = NUM_NEIGHBORS):
    """Find the K nearest neighbors of a neuron in node2vec embedding space.

    :param node_id: root ID of the query neuron (must be in the hub subgraph).
    :param k: number of neighbors to return.
    :returns: tuple of (list of neighbor node IDs, list of cosine
        similarities), both sorted by similarity descending.
    """
    query_row = _node_id_to_row[node_id]
    query_vector = _vectors[query_row]

    norms = np.linalg.norm(_vectors, axis=1) * np.linalg.norm(query_vector)
    similarities = (_vectors @ query_vector) / np.where(norms == 0, 1, norms)
    similarities[query_row] = -np.inf  # exclude the query neuron itself

    top_k = np.argsort(similarities)[::-1][:k]
    return [int(_node_ids[i]) for i in top_k], [float(similarities[i]) for i in top_k]


def search_neighbors(neuron_id_text: str, color_by: str, projection: str):
    """Gradio callback: look up a neuron and return its highlighted plot + neighbor table.

    :param neuron_id_text: neuron root ID typed into the search box.
    :param color_by: current color-by selection, forwarded to the plot.
    :param projection: current projection selection, forwarded to the plot.
    :returns: tuple of (Plotly figure, neighbor DataFrame or an empty
        DataFrame with a message if the ID wasn't found).
    """
    fig = build_projection_figure(color_by=color_by, projection=projection, highlight_id=neuron_id_text)

    highlight_ids = _resolve_highlight_ids(neuron_id_text)
    if not highlight_ids:
        message = "Enter a numeric neuron (root) ID present in the hub subgraph." if neuron_id_text.strip() else ""
        return fig, pd.DataFrame({"message": [message]} if message else {})

    node_id = highlight_ids[0]
    neighbor_ids, similarities = nearest_neighbors(node_id, k=NUM_NEIGHBORS)
    table = _embeddings_df.loc[neighbor_ids, ["cell_type", "super_class", "primary_neuropil"]].reset_index()
    table = table.rename(columns={"index": "node_id"})
    table.insert(1, "cosine_similarity", [round(s, 3) for s in similarities])
    return fig, table


def build() -> None:
    """Add the Neuron Embeddings tab's components to the enclosing ``gr.Blocks``.

    Must be called inside an active ``gr.Blocks()`` context, e.g. from
    ``app.py``.
    """
    with gr.Tab("Neuron Embeddings"):
        gr.Markdown(
            "### Connectivity-based neuron embeddings\n"
            "Node2vec embeddings trained only on wiring topology (which neurons "
            "connect to which), projected to 2D."
        )
        gr.Markdown(_agreement_caption())

        with gr.Row():
            color_by = gr.Dropdown(COLOR_BY_CHOICES, value="super_class", label="Color by")
            projection = gr.Radio(PROJECTION_CHOICES, value="pca", label="Projection")

        plot = gr.Plot(build_projection_figure())

        with gr.Row():
            search_box = gr.Textbox(label="Search by neuron (root) ID", placeholder="e.g. 720575940613635737")
            search_button = gr.Button("Highlight nearest neighbors")

        neighbor_table = gr.Dataframe(label="Nearest neighbors in embedding space")

        def _refresh_plot(color_by_value, projection_value, neuron_id_text):
            return build_projection_figure(color_by_value, projection_value, neuron_id_text)

        color_by.change(_refresh_plot, [color_by, projection, search_box], plot)
        projection.change(_refresh_plot, [color_by, projection, search_box], plot)
        search_button.click(search_neighbors, [search_box, color_by, projection], [plot, neighbor_table])
        search_box.submit(search_neighbors, [search_box, color_by, projection], [plot, neighbor_table])
