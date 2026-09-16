# /data

This directory holds generated and downloaded artifacts. Everything here except
this file is gitignored — nothing in `/data` is committed to the repository.

## What gets written here

Populated by the scripts in `/ingest`, in order:

- `raw/` — synapse table and edge list downloaded from the Zenodo static
  release referenced in `flyconnectome/flywire_annotations`
  (https://zenodo.org/records/10676866), plus any cell-type annotation
  tables. Produced by `ingest/fetch_connectome.py`.
- `graph.gpickle` (or `graph.graphml`) — the constructed NetworkX graph with
  neuron metadata attached to nodes. Produced by `ingest/build_graph.py`.
- `graph_stats.parquet` — per-node degree/betweenness centrality and detected
  community assignments. Produced by `ingest/compute_graph_stats.py`.
- `embeddings.parquet` — node2vec embeddings and their 2D projection
  (PCA/UMAP) per neuron. Produced by `ingest/compute_embeddings.py`.

## Regenerating

Run the `/ingest` scripts in order (each caches its own output, so re-running
an earlier step invalidates everything downstream):

```bash
python ingest/fetch_connectome.py
python ingest/build_graph.py
python ingest/compute_graph_stats.py
python ingest/compute_embeddings.py
```

The Gradio app in `/app` only reads the cached parquet/pickle files above — it
never re-runs this pipeline itself.
