"""Compute structural statistics over the connectome hub subgraph.

Loads ``data/graph.gpickle`` (built by ``build_graph.py``), computes
per-node degree and betweenness centrality plus Louvain community
assignments, and writes:

- ``data/graph_stats.parquet`` - one row per node, with degree/centrality/
  community columns alongside its cell-type metadata, for the leaderboard
  and community-colored visualization in the Graph Theory tab.
- ``data/graph_summary.json`` - graph-level stats (node/edge counts,
  density, average shortest path length) for the tab's stats panel.

Also precomputes a 2D spring layout for every node, so the Graph Theory
tab's interactive visualization only has to plot cached coordinates rather
than run graph layout at request time.

Betweenness centrality and average shortest path length are computed on the
unweighted topology (standard hop-count shortest paths), matching the usual
graph-theory definitions of these metrics. Louvain community detection uses
raw ``syn_count`` as edge weight, since modularity optimization expects
higher weight to mean a *stronger* tie between two neurons.
"""

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import GRAPH_PATH, NODE_METADATA_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH_STATS_PATH = DATA_DIR / "graph_stats.parquet"
GRAPH_SUMMARY_PATH = DATA_DIR / "graph_summary.json"


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    """Load the hub subgraph pickled by ``build_graph.py``.

    :param path: path to the pickled graph.
    :returns: the deserialized directed graph.
    """
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def compute_node_stats(graph: nx.DiGraph) -> pd.DataFrame:
    """Compute per-node degree, centrality, and community assignment.

    :param graph: hub subgraph, as built by ``build_graph.py``.
    :returns: DataFrame indexed by node ID with columns ``in_degree``,
        ``out_degree``, ``total_synapse_degree``, ``betweenness``,
        ``community``, ``layout_x``/``layout_y`` (precomputed spring
        layout position), and the node metadata columns in
        :data:`build_graph.NODE_METADATA_COLUMNS`.
    """
    betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)

    undirected = graph.to_undirected()
    communities = nx.algorithms.community.louvain_communities(undirected, weight="syn_count", seed=42)
    community_of = {}
    for community_id, members in enumerate(communities):
        for node_id in members:
            community_of[node_id] = community_id

    layout = nx.spring_layout(undirected, weight="syn_count", seed=42)

    rows = []
    for node_id, data in graph.nodes(data=True):
        row = {
            "node_id": node_id,
            "in_degree": graph.in_degree(node_id),
            "out_degree": graph.out_degree(node_id),
            "total_synapse_degree": data.get("total_synapse_degree"),
            "betweenness": betweenness[node_id],
            "community": community_of[node_id],
            "primary_neuropil": data.get("primary_neuropil"),
            "layout_x": layout[node_id][0],
            "layout_y": layout[node_id][1],
        }
        for col in NODE_METADATA_COLUMNS:
            row[col] = data.get(col)
        rows.append(row)

    return pd.DataFrame(rows).set_index("node_id")


def compute_graph_summary(graph: nx.DiGraph) -> dict:
    """Compute graph-level summary statistics for the stats panel.

    :param graph: hub subgraph, as built by ``build_graph.py``.
    :returns: dict with ``num_nodes``, ``num_edges``, ``density``,
        ``avg_shortest_path_length``, and ``num_communities``.
    """
    undirected = graph.to_undirected()
    avg_path_length = nx.average_shortest_path_length(undirected, weight=None)
    communities = nx.algorithms.community.louvain_communities(undirected, weight="syn_count", seed=42)

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_shortest_path_length": avg_path_length,
        "num_communities": len(communities),
    }


if __name__ == "__main__":
    g = load_graph()

    stats_df = compute_node_stats(g)
    GRAPH_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(GRAPH_STATS_PATH)

    summary = compute_graph_summary(g)
    with open(GRAPH_SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Node stats saved to {GRAPH_STATS_PATH} ({len(stats_df)} rows)")
    print("Graph summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
