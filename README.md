# Fly Connectome Explorer

A Gradio app for exploring the **BANC (Brain-And-Nerve-Cord)** *Drosophila*
connectome — the full fly brain + ventral nerve cord dataset (166K neurons,
125M synapses, CC-BY licensed) — through two lenses: classical graph theory
and connectivity-based neuron embeddings.

## Tabs

1. **Graph Theory** — degree distribution, a hub-neuron centrality
   leaderboard, an interactive subgraph visualization colored by detected
   community, and summary stats (node/edge counts, density, average path
   length) on the analyzed subgraph.
2. **Neuron Embeddings** — a 2D projection of node2vec embeddings colored by
   known cell type/region, a neuron-ID search box that highlights nearest
   neighbors in embedding space, and a note on how structural similarity
   relates to functional similarity.

## Data source

Connectome data is pulled from the static Zenodo release referenced by
[`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations)
([record 10676866](https://zenodo.org/records/10676866)) — synapse table,
edge list, and cell-type annotations as flat files. This avoids live
FlyWire/CAVE API auth and gives reproducible, cacheable input files.

Live queries via `fafbseg-py` + CAVE (token from
[global.daf-apis.com](https://global.daf-apis.com/auth/api/v1/user/token),
read from the `FLYWIRE_TOKEN` environment variable — never hardcoded) are
used only as a fallback for data not present in the static files.

## Scale

125M synapses is too large to hold or query live in a Space. The `/ingest`
pipeline builds the full graph once, subsamples it to a tractable, connected
subgraph, and precomputes centrality, community detection, and node2vec
embeddings to `/data` as parquet/pickle. The specific subsampling strategy
and its rationale are documented in `/ingest/build_graph.py` once implemented.
The Gradio app in `/app` only loads these cached artifacts — it never
recomputes them at startup or per-request.

## Repository structure

```
/ingest
  fetch_connectome.py     - downloads/loads synapse + edge data
  build_graph.py           - constructs NetworkX graph with neuron metadata
  compute_graph_stats.py   - degree/betweenness centrality, Louvain communities
  compute_embeddings.py    - node2vec embeddings + 2D projection (PCA/UMAP)
/app
  app.py                   - Gradio interface, two tabs, loads cached artifacts only
  graph_tab.py
  embeddings_tab.py
/data                      - generated artifacts (gitignored; see data/README.md)
requirements.txt
```

## Running the pipeline

```bash
pip install -r requirements.txt
python ingest/fetch_connectome.py
python ingest/build_graph.py
python ingest/compute_graph_stats.py
python ingest/compute_embeddings.py
python app/app.py
```

## About / Ethics

This project performs **structural graph analysis only**. Node2vec
embeddings and graph statistics describe wiring topology — they are **not**
a model of behavior, cognition, comprehension, or any cognitive process, and
no claims of that kind should be drawn from this tool. It is intended as an
exploratory and educational aid for looking at connectome structure.

Data: [BANC / FlyWire connectome](https://flywire.ai/), CC-BY licensed.
See the [flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
repository and its associated paper for methodology and provenance.
