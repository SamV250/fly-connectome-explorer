---
title: Fly Connectome Explorer
emoji: 🪰
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
app_file: app/app.py
pinned: false
---

# Fly Connectome Explorer

A Gradio app for exploring the **FlyWire/BANC** *Drosophila* connectome
through two lenses: classical graph theory and connectivity-based neuron
embeddings.

**Data note:** the ingest pipeline pulls from the FlyWire whole-brain
connectome release (Zenodo record 10676866 — 139K proofread neurons, ~130M
synapses total, CC-BY licensed). This is the female adult fly brain (FAFB),
not a combined brain+VNC BANC release; see [Data source](#data-source) below.

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
([record 10676866](https://zenodo.org/records/10676866)):
`proofread_connections_783.feather` (the proofread neuron-to-neuron
connection table, already aggregated per synapse count — used instead of
the 9.4 GB raw synapse table) and `proofread_root_ids_783.npy`, plus the
neuron cell-type/class annotation table from that same GitHub repository.
This avoids live FlyWire/CAVE API auth and gives reproducible, cacheable
input files.

Live queries via `fafbseg-py` + CAVE (token from
[global.daf-apis.com](https://global.daf-apis.com/auth/api/v1/user/token),
read from the `FLYWIRE_TOKEN` environment variable — never hardcoded) are
used only as a fallback for data not present in the static files; this
fallback path is documented but not implemented, since the static files
have covered every need so far.

## Scale and subsampling

The proofread connectome has ~139K neurons and ~15M unique connections —
too large to hold or query live in a Space. `ingest/build_graph.py` ranks
every neuron by total synaptic degree (computed over the *full* network)
and keeps the induced subgraph on the **top 1,000 hub neurons**. This
subgraph comes out fully connected (a single component, no fragments to
discard) at 1,000 nodes / 53,362 edges, dense enough for centrality,
community detection, and node2vec to show non-trivial structure, and small
enough that every step of the pipeline finishes in seconds to low minutes.
A single-brain-region sample was considered and rejected: inter-region
connections are common enough that restricting to one neuropil tends to
fragment the induced subgraph into disconnected pieces.

`/ingest` precomputes centrality, Louvain communities, a network layout,
and node2vec embeddings once, caching everything to `/data` as
parquet/pickle/JSON. The Gradio app in `/app` only loads these cached
artifacts — it never recomputes them at startup or per-request.

## Repository structure

```
/ingest
  fetch_connectome.py     - downloads/caches the connection table, root IDs, and annotations
  build_graph.py           - builds the hub-neuron NetworkX graph with neuron metadata
  compute_graph_stats.py   - degree/betweenness centrality, Louvain communities, layout
  compute_embeddings.py    - node2vec embeddings + 2D projection (PCA/UMAP) + KMeans
/app
  app.py                   - Gradio interface, two tabs, loads cached artifacts only
  graph_tab.py
  embeddings_tab.py
/data                      - generated artifacts (gitignored; see data/README.md)
requirements.txt
```

## Running the pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ingest/fetch_connectome.py      # downloads ~880 MB to data/raw/
python ingest/build_graph.py
python ingest/compute_graph_stats.py
python ingest/compute_embeddings.py    # trains node2vec, ~2 minutes

python app/app.py
```

## About / Ethics

This project performs **structural graph analysis only**. Node2vec
embeddings and graph statistics describe wiring topology — they are **not**
a model of behavior, cognition, comprehension, or any cognitive process, and
no claims of that kind should be drawn from this tool. It is intended as an
exploratory and educational aid for looking at connectome structure.

Data: [FlyWire/BANC connectome](https://flywire.ai/), CC-BY licensed.
See the [flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
repository and its associated paper for methodology and provenance.
