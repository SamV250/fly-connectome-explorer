"""Construct a NetworkX graph of the connectome with neuron metadata.

Reads the proofread connection table and cell-type annotations cached by
``fetch_connectome.py``, builds a directed, synapse-weighted
``networkx.DiGraph``, subsamples it down to a tractable hub-neuron subgraph,
and writes the result to ``data/graph.gpickle``.

Subsampling strategy
--------------------
The full proofread network has ~139K neurons and ~15M unique (pre, post)
connections - far too large to hold interactively in a Gradio Space or run
betweenness centrality / node2vec on within a request cycle. Instead we keep
the ``TOP_N_HUBS`` neurons with the highest total synaptic degree (in-degree
+ out-degree, weighted by synapse count) computed over the *full* network,
then take the subgraph induced by just those neurons.

This "hub subgraph" approach was chosen over sampling a single brain region
because hub neurons are, by construction, densely interconnected: at
``TOP_N_HUBS = 1000`` the induced subgraph is a single connected component
(no isolated fragments to discard or explain away), dense enough for
betweenness centrality/community detection/node2vec to produce meaningful,
non-degenerate structure, and small enough (~1000 nodes, ~53K edges) that
every downstream step in this pipeline finishes in seconds. A single-region
sample was rejected because inter-region connections are common enough that
restricting to one neuropil tends to fragment the induced subgraph into many
disconnected pieces.
"""

from pathlib import Path
import pickle

import networkx as nx
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONNECTIONS_PATH = RAW_DIR / "proofread_connections_783.feather"
ANNOTATIONS_PATH = RAW_DIR / "flywire_neuron_annotations.tsv"
GRAPH_PATH = DATA_DIR / "graph.gpickle"

# Number of highest-total-degree neurons to keep. See module docstring for
# why this size was chosen.
TOP_N_HUBS = 1000

NODE_METADATA_COLUMNS = [
    "cell_type",
    "hemibrain_type",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "side",
    "nerve",
    "top_nt",
    "pos_x",
    "pos_y",
    "pos_z",
]


def load_edge_table(connections_path: Path = CONNECTIONS_PATH) -> pd.DataFrame:
    """Load and aggregate the proofread connection table into unique directed edges.

    The raw table has one row per (pre neuron, post neuron, neuropil) triple;
    this collapses it to one row per (pre neuron, post neuron) pair by
    summing synapse counts across neuropils, since edge weight in the graph
    represents total synaptic strength between two neurons regardless of
    where in the brain that connection is made.

    :param connections_path: path to ``proofread_connections_783.feather``.
    :returns: DataFrame with columns ``pre_pt_root_id``, ``post_pt_root_id``,
        ``syn_count``.
    """
    df = pd.read_feather(connections_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    return df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()


def select_hub_neurons(edges: pd.DataFrame, top_n: int = TOP_N_HUBS) -> pd.Index:
    """Rank neurons by total synaptic degree and return the top N.

    :param edges: aggregated edge table, as returned by :func:`load_edge_table`.
    :param top_n: number of highest-degree neurons to keep.
    :returns: index of the ``top_n`` neuron root IDs, by total (in + out)
        synapse count computed over the full (unsampled) network.
    """
    out_degree = edges.groupby("pre_pt_root_id")["syn_count"].sum()
    in_degree = edges.groupby("post_pt_root_id")["syn_count"].sum()
    total_degree = out_degree.add(in_degree, fill_value=0).sort_values(ascending=False)
    return total_degree.head(top_n).index


def load_node_metadata(annotations_path: Path = ANNOTATIONS_PATH) -> pd.DataFrame:
    """Load per-neuron cell-type/class annotations, indexed by root ID.

    :param annotations_path: path to the flywire_annotations neuron TSV.
    :returns: DataFrame indexed by ``root_id`` with the columns in
        :data:`NODE_METADATA_COLUMNS`.
    """
    ann = pd.read_csv(
        annotations_path,
        sep="\t",
        usecols=["root_id"] + NODE_METADATA_COLUMNS,
        low_memory=False,
    )
    ann = ann.drop_duplicates(subset="root_id", keep="first")
    return ann.set_index("root_id")


def compute_primary_neuropil(connections_path: Path, neuron_ids: set) -> pd.Series:
    """Determine each neuron's dominant neuropil by total synapse count.

    Sums synapse counts per (neuron, neuropil) across both the pre- and
    post-synaptic role, then takes the neuropil with the largest total for
    each neuron. Used as a coarse anatomical "region" label for nodes that
    lack a ``cell_type``/``super_class`` annotation.

    :param connections_path: path to ``proofread_connections_783.feather``.
    :param neuron_ids: set of root IDs to compute this for (the hub subgraph).
    :returns: Series mapping root ID -> dominant neuropil name.
    """
    df = pd.read_feather(
        connections_path, columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"]
    )
    df = df[df["pre_pt_root_id"].isin(neuron_ids) | df["post_pt_root_id"].isin(neuron_ids)]

    pre = df[df["pre_pt_root_id"].isin(neuron_ids)][["pre_pt_root_id", "neuropil", "syn_count"]]
    pre = pre.rename(columns={"pre_pt_root_id": "root_id"})
    post = df[df["post_pt_root_id"].isin(neuron_ids)][["post_pt_root_id", "neuropil", "syn_count"]]
    post = post.rename(columns={"post_pt_root_id": "root_id"})

    combined = pd.concat([pre, post], ignore_index=True)
    totals = combined.groupby(["root_id", "neuropil"])["syn_count"].sum()
    return totals.groupby("root_id").idxmax().apply(lambda pair: pair[1])


def build_graph(top_n: int = TOP_N_HUBS) -> nx.DiGraph:
    """Build the hub-neuron connectome subgraph with node/edge metadata.

    :param top_n: number of highest-degree neurons to keep (see
        :func:`select_hub_neurons`).
    :returns: a directed graph where nodes are neuron root IDs (int) with
        metadata attributes, and edges carry ``syn_count`` (total synapses
        between that pair).
    """
    edges = load_edge_table()
    hub_ids = select_hub_neurons(edges, top_n=top_n)
    hub_id_set = set(hub_ids)

    sub_edges = edges[
        edges["pre_pt_root_id"].isin(hub_id_set) & edges["post_pt_root_id"].isin(hub_id_set)
    ]

    graph = nx.from_pandas_edgelist(
        sub_edges,
        source="pre_pt_root_id",
        target="post_pt_root_id",
        edge_attr="syn_count",
        create_using=nx.DiGraph,
    )

    # Guarantee the graph the app ships with is a single connected piece.
    largest_component = max(nx.weakly_connected_components(graph), key=len)
    if len(largest_component) < graph.number_of_nodes():
        graph = graph.subgraph(largest_component).copy()

    total_degree = edges.groupby("pre_pt_root_id")["syn_count"].sum().add(
        edges.groupby("post_pt_root_id")["syn_count"].sum(), fill_value=0
    )
    node_meta = load_node_metadata()
    primary_neuropil = compute_primary_neuropil(CONNECTIONS_PATH, set(graph.nodes))

    for node_id in graph.nodes:
        graph.nodes[node_id]["total_synapse_degree"] = float(total_degree.get(node_id, 0))
        graph.nodes[node_id]["primary_neuropil"] = primary_neuropil.get(node_id)
        if node_id in node_meta.index:
            row = node_meta.loc[node_id]
            for col in NODE_METADATA_COLUMNS:
                value = row[col]
                graph.nodes[node_id][col] = None if pd.isna(value) else value
        else:
            for col in NODE_METADATA_COLUMNS:
                graph.nodes[node_id][col] = None

    graph.graph["top_n_hubs"] = top_n
    return graph


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    """Pickle a graph to disk, creating the parent directory if needed.

    :param graph: graph to serialize.
    :param path: destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    g = build_graph()
    save_graph(g)
    print(f"Built graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Saved to {GRAPH_PATH} ({GRAPH_PATH.stat().st_size / 1e6:.2f} MB)")
