"""Compute connectivity-based neuron embeddings.

Loads ``data/graph.gpickle`` (the hub-neuron subgraph built by
``build_graph.py``), trains node2vec embeddings over its topology, projects
them to 2D with both PCA and UMAP, and clusters them with KMeans so the
Neuron Embeddings tab can compare structural clusters against known
cell-type/region labels. Writes one row per neuron to
``data/embeddings.parquet``.
"""

import json
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from build_graph import GRAPH_PATH, NODE_METADATA_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.parquet"

# node2vec hyperparameters. Chosen to be fast (~seconds) on a ~1000-node
# hub subgraph while still giving walks enough length/breadth to capture
# multi-hop connectivity structure.
EMBEDDING_DIM = 64
WALK_LENGTH = 30
NUM_WALKS = 200
WORKERS = 4


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    """Load the hub subgraph pickled by ``build_graph.py``.

    :param path: path to the pickled graph.
    :returns: the deserialized directed graph.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


def train_node2vec(graph: nx.DiGraph) -> dict:
    """Train node2vec embeddings over the graph's (undirected) topology.

    Edge weights (``syn_count``) bias random walks toward strongly
    connected neuron pairs. The graph is treated as undirected for
    walk generation, since embedding similarity is meant to capture
    "who talks to whom", not the direction of signal flow.

    :param graph: hub subgraph, as built by ``build_graph.py``.
    :returns: mapping of node ID (int) -> embedding vector (``np.ndarray``
        of shape ``(EMBEDDING_DIM,)``).
    """
    undirected = graph.to_undirected()
    node2vec = Node2Vec(
        undirected,
        dimensions=EMBEDDING_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        weight_key="syn_count",
        workers=WORKERS,
        seed=42,
        quiet=True,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=64)
    return {node_id: model.wv[str(node_id)] for node_id in undirected.nodes}


def project_and_cluster(embeddings: dict, num_clusters: int) -> pd.DataFrame:
    """Project embeddings to 2D (PCA, UMAP) and cluster them (KMeans).

    :param embeddings: mapping of node ID -> embedding vector.
    :param num_clusters: number of KMeans clusters to fit.
    :returns: DataFrame indexed by node ID with columns ``embedding``
        (list of floats), ``pca_x``, ``pca_y``, ``umap_x``, ``umap_y``, and
        ``embedding_cluster``.
    """
    node_ids = list(embeddings.keys())
    vectors = np.stack([embeddings[node_id] for node_id in node_ids])

    pca_coords = PCA(n_components=2, random_state=42).fit_transform(vectors)

    try:
        import umap

        umap_coords = umap.UMAP(n_components=2, random_state=42).fit_transform(vectors)
    except ImportError:
        umap_coords = np.full((len(node_ids), 2), np.nan)

    clusters = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(vectors)

    return pd.DataFrame(
        {
            "node_id": node_ids,
            "embedding": list(vectors),
            "pca_x": pca_coords[:, 0],
            "pca_y": pca_coords[:, 1],
            "umap_x": umap_coords[:, 0],
            "umap_y": umap_coords[:, 1],
            "embedding_cluster": clusters,
        }
    ).set_index("node_id")


def attach_metadata(embeddings_df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    """Copy node cell-type metadata from the graph onto the embeddings table.

    :param embeddings_df: output of :func:`project_and_cluster`.
    :param graph: hub subgraph carrying the node metadata attributes.
    :returns: ``embeddings_df`` with metadata columns added.
    """
    for col in NODE_METADATA_COLUMNS + ["primary_neuropil", "total_synapse_degree"]:
        embeddings_df[col] = [graph.nodes[node_id].get(col) for node_id in embeddings_df.index]
    return embeddings_df


if __name__ == "__main__":
    g = load_graph()

    start = time.time()
    node_embeddings = train_node2vec(g)
    train_time = time.time() - start

    num_clusters = len({data.get("super_class") for _, data in g.nodes(data=True) if data.get("super_class")})
    df = project_and_cluster(node_embeddings, num_clusters=num_clusters)
    df = attach_metadata(df, g)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EMBEDDINGS_PATH)

    print(f"Trained node2vec in {train_time:.1f}s: {len(node_embeddings)} nodes, dim={EMBEDDING_DIM}")
    print(f"KMeans clusters: {num_clusters} (matching number of distinct super_class labels)")
    print(f"Saved to {EMBEDDINGS_PATH} ({EMBEDDINGS_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {df.shape}, columns: {list(df.columns)}")
