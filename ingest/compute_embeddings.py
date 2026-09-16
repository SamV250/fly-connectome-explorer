"""Compute connectivity-based neuron embeddings.

Loads ``data/graph.gpickle``, trains node2vec embeddings over the graph
topology, projects them to 2D with PCA (or UMAP if available), and writes
per-neuron embedding vectors and 2D coordinates to ``data/embeddings.parquet``.

Implemented in a later commit.
"""

if __name__ == "__main__":
    raise NotImplementedError("compute_embeddings.py: implemented in step 4")
