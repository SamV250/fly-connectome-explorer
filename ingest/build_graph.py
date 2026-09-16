"""Construct a NetworkX graph of the connectome with neuron metadata.

Reads the raw synapse/edge tables cached by ``fetch_connectome.py``, builds a
directed, weighted ``networkx.DiGraph`` (nodes = neurons with cell-type/region
metadata, edges = synaptic connections weighted by synapse count), subsamples
to a tractable size, and writes the result to ``data/graph.gpickle``.

Implemented in a later commit.
"""

if __name__ == "__main__":
    raise NotImplementedError("build_graph.py: implemented in step 3")
