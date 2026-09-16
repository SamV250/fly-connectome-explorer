"""Graph Theory tab for the Gradio app.

Renders degree distribution, a hub-neuron centrality leaderboard, an
interactive subgraph visualization colored by detected community, and a
summary stats panel - all sourced from ``data/graph_stats.parquet``,
``data/graph_summary.json``, and ``data/graph.gpickle``. Loads its cached
artifacts once at import time; every callback below only re-reads
in-memory DataFrames/figures, no graph algorithms run per request.
"""

import json
import pickle
from pathlib import Path

import gradio as gr
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Cap how many edges get drawn in the network plot: with ~53K edges among
# 1000 nodes, drawing all of them makes an illegible black smear. Showing
# only the strongest ties keeps the plot responsive and readable while
# still conveying the backbone structure that the community colors sit on.
EDGE_DISPLAY_LIMIT = 2500

LEADERBOARD_SIZE = 25

_stats_df = pd.read_parquet(DATA_DIR / "graph_stats.parquet")
with open(DATA_DIR / "graph_summary.json") as _fh:
    _summary = json.load(_fh)
with open(DATA_DIR / "graph.gpickle", "rb") as _fh:
    _graph: nx.DiGraph = pickle.load(_fh)

_COMMUNITY_COLORS = px.colors.qualitative.Bold


def _community_color(community_id: int) -> str:
    """Map a community ID to a stable color from a qualitative palette.

    :param community_id: Louvain community index.
    :returns: a CSS color string.
    """
    return _COMMUNITY_COLORS[community_id % len(_COMMUNITY_COLORS)]


def build_stats_panel_markdown() -> str:
    """Render the graph-level summary stats as a Markdown panel.

    :returns: Markdown text with node/edge counts, density, average
        shortest path length, and community count.
    """
    return (
        f"**Nodes:** {_summary['num_nodes']:,}  \n"
        f"**Edges:** {_summary['num_edges']:,}  \n"
        f"**Density:** {_summary['density']:.4f}  \n"
        f"**Avg. shortest path length:** {_summary['avg_shortest_path_length']:.2f} hops  \n"
        f"**Communities detected:** {_summary['num_communities']}"
    )


def build_degree_distribution_figure() -> go.Figure:
    """Build a histogram of total synaptic degree across hub neurons.

    :returns: a Plotly histogram figure (log-scaled x-axis, since synaptic
        degree is heavily right-skewed even within this hub subgraph).
    """
    fig = px.histogram(
        _stats_df,
        x="total_synapse_degree",
        nbins=40,
        labels={"total_synapse_degree": "Total synaptic degree (in + out)"},
        title="Degree distribution (hub subgraph)",
    )
    fig.update_xaxes(type="log")
    fig.update_layout(yaxis_title="Number of neurons", margin=dict(t=40))
    return fig


def build_leaderboard_dataframe() -> pd.DataFrame:
    """Build the top-N hub neuron leaderboard, ranked by betweenness centrality.

    :returns: DataFrame with node ID, cell type, community, and centrality
        columns, sorted by betweenness centrality descending.
    """
    columns = [
        "in_degree",
        "out_degree",
        "total_synapse_degree",
        "betweenness",
        "community",
        "cell_type",
        "super_class",
        "primary_neuropil",
    ]
    top = _stats_df.sort_values("betweenness", ascending=False).head(LEADERBOARD_SIZE)
    return top[columns].reset_index().rename(columns={"index": "node_id"})


def build_network_figure() -> go.Figure:
    """Build an interactive subgraph visualization colored by community.

    Node positions come from the spring layout precomputed in
    ``compute_graph_stats.py``. Only the ``EDGE_DISPLAY_LIMIT`` strongest
    edges (by synapse count) are drawn, to keep the plot legible.

    :returns: a Plotly figure with edge lines and community-colored node
        markers.
    """
    edges = sorted(_graph.edges(data=True), key=lambda e: e[2]["syn_count"], reverse=True)
    edges = edges[:EDGE_DISPLAY_LIMIT]

    edge_x, edge_y = [], []
    for source, target, _ in edges:
        edge_x += [_stats_df.loc[source, "layout_x"], _stats_df.loc[target, "layout_x"], None]
        edge_y += [_stats_df.loc[source, "layout_y"], _stats_df.loc[target, "layout_y"], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    for community_id, group in _stats_df.groupby("community"):
        hover_text = [
            f"{node_id}<br>{row.cell_type or 'unknown'} ({row.super_class or 'unknown'})"
            for node_id, row in group.iterrows()
        ]
        fig.add_trace(
            go.Scattergl(
                x=group["layout_x"],
                y=group["layout_y"],
                mode="markers",
                name=f"Community {community_id}",
                text=hover_text,
                hoverinfo="text",
                marker=dict(size=7, color=_community_color(community_id)),
            )
        )

    fig.update_layout(
        title=f"Hub subgraph, top {len(edges):,} edges by synapse count, colored by community",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40),
    )
    return fig


def build() -> None:
    """Add the Graph Theory tab's components to the enclosing ``gr.Blocks``.

    Must be called inside an active ``gr.Blocks()`` context, e.g. from
    ``app.py``.
    """
    with gr.Tab("Graph Theory"):
        gr.Markdown(
            "### Structural analysis of the hub-neuron connectome subgraph\n"
            "The 1,000 neurons with the highest total synaptic degree, and every "
            "connection between them."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(build_stats_panel_markdown())
            with gr.Column(scale=2):
                gr.Plot(build_degree_distribution_figure())

        gr.Plot(build_network_figure())

        gr.Markdown(f"#### Top {LEADERBOARD_SIZE} neurons by betweenness centrality")
        gr.Dataframe(build_leaderboard_dataframe())
