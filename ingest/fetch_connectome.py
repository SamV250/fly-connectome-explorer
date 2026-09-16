"""Download and cache the BANC/FlyWire connectome source tables.

Fetches the static synapse table / edge list and cell-type annotations from
the Zenodo record backing ``flyconnectome/flywire_annotations``
(https://zenodo.org/records/10676866), and writes them to ``data/raw/``.
Falls back to ``fafbseg`` + CAVE token auth (``FLYWIRE_TOKEN`` env var) only
when the static files don't contain what's needed.

Implemented in a later commit.
"""

if __name__ == "__main__":
    raise NotImplementedError("fetch_connectome.py: implemented in step 2")
