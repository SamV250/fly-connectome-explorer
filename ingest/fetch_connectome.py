"""Download and cache the FlyWire/BANC connectome source tables.

Pulls the proofread neuron-to-neuron connection table and the list of
proofread root IDs from the static Zenodo release backing
``flyconnectome/flywire_annotations`` (record 10676866:
https://zenodo.org/records/10676866), plus the neuron cell-type/class
annotation table from that same GitHub repository. All downloads are cached
to ``data/raw/`` and skipped on subsequent runs if already present and
checksum-valid.

This avoids querying the full ~130M-row raw synapse table
(``flywire_synapses_783.feather``, ~9.4 GB): ``proofread_connections_783.feather``
is already synapse counts pre-aggregated per (pre neuron, post neuron,
neuropil) pair, which is exactly the edge list this project needs, at a
fraction of the size (~850 MB).

If a future analysis needs data not present in these static files, fall back
to ``fafbseg`` + CAVE token auth (see :func:`fetch_via_cave_fallback`), never
by hardcoding credentials — the token is read from the ``FLYWIRE_TOKEN``
environment variable.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests

ZENODO_RECORD_ID = "10676866"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Files pulled from the Zenodo record. proofread_connections is the edge
# list; proofread_root_ids is the authoritative set of valid (proofread)
# neuron IDs used to sanity-check the graph.
ZENODO_FILES = [
    "proofread_connections_783.feather",
    "proofread_root_ids_783.npy",
]

# Neuron cell-type/class annotations, from the companion GitHub repo. Small
# flat file, not part of the Zenodo record itself.
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
ANNOTATIONS_FILENAME = "flywire_neuron_annotations.tsv"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK_SIZE = 1 << 20  # 1 MiB


def _md5sum(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the MD5 checksum of a file.

    :param path: file to hash.
    :param chunk_size: bytes to read per iteration.
    :returns: hex-encoded MD5 digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    dest: Path,
    expected_md5: Optional[str] = None,
    timeout: int = 60,
) -> Path:
    """Stream a URL to disk, skipping the download if a valid file already exists.

    :param url: source URL to GET.
    :param dest: destination file path.
    :param expected_md5: if given, the download is verified against this MD5
        checksum (and skipped early if ``dest`` already matches it).
    :param timeout: per-request timeout in seconds, passed to ``requests``.
    :returns: path to the downloaded (or already-cached) file.
    :raises requests.HTTPError: if the download request fails.
    :raises ValueError: if the downloaded file does not match ``expected_md5``.
    """
    if dest.exists() and expected_md5 is not None and _md5sum(dest) == expected_md5:
        print(f"[skip] {dest.name} already cached and checksum-valid")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    print(f"[fetch] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(part_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="")
        print()

    if expected_md5 is not None:
        actual = _md5sum(part_path)
        if actual != expected_md5:
            part_path.unlink(missing_ok=True)
            raise ValueError(
                f"checksum mismatch for {dest.name}: expected {expected_md5}, got {actual}"
            )

    part_path.replace(dest)
    return dest


def fetch_zenodo_file_index(record_id: str = ZENODO_RECORD_ID) -> dict:
    """Fetch the Zenodo record's file listing (name -> {url, checksum, size}).

    :param record_id: Zenodo record ID to query.
    :returns: mapping of file key to a dict with ``url``, ``checksum`` (MD5,
        without the ``md5:`` prefix), and ``size`` (bytes).
    :raises requests.HTTPError: if the Zenodo API request fails.
    """
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    response.raise_for_status()
    record = response.json()

    index = {}
    for file_entry in record.get("files", []):
        checksum = file_entry.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        index[file_entry["key"]] = {
            "url": file_entry["links"]["self"],
            "checksum": md5,
            "size": file_entry.get("size"),
        }
    return index


def fetch_via_cave_fallback() -> None:
    """Fetch connectome data live via ``fafbseg`` + CAVE, as a fallback.

    Only used when the static Zenodo files don't contain something an
    analysis needs. Requires a CAVE auth token in the ``FLYWIRE_TOKEN``
    environment variable (obtained from
    https://global.daf-apis.com/auth/api/v1/user/token); the token is never
    hardcoded or read from any other source.

    :raises RuntimeError: always, until a concrete live-query need arises —
        this function is intentionally not implemented until one does, to
        avoid unused CAVE auth code paths.
    """
    token = os.environ.get("FLYWIRE_TOKEN")
    if not token:
        raise RuntimeError(
            "FLYWIRE_TOKEN environment variable is not set; obtain a token from "
            "https://global.daf-apis.com/auth/api/v1/user/token"
        )
    raise RuntimeError(
        "CAVE fallback is not implemented: the static Zenodo files "
        "(proofread_connections_783.feather) have covered every need so far."
    )


def fetch_connectome(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Download the connectome edge list, valid neuron IDs, and cell-type annotations.

    :param data_dir: directory to cache downloads in (created if missing).
    :returns: mapping of logical name (``"connections"``, ``"root_ids"``,
        ``"annotations"``) to the local file path.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_index = fetch_zenodo_file_index()

    paths = {}
    for key in ZENODO_FILES:
        entry = file_index[key]
        dest = data_dir / key
        _download(entry["url"], dest, expected_md5=entry["checksum"])
        logical_name = "connections" if "connections" in key else "root_ids"
        paths[logical_name] = dest

    annotations_dest = data_dir / ANNOTATIONS_FILENAME
    _download(ANNOTATIONS_URL, annotations_dest)
    paths["annotations"] = annotations_dest

    return paths


if __name__ == "__main__":
    downloaded = fetch_connectome()
    print("\nDownloaded:")
    for name, path in downloaded.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {path} ({size_mb:.1f} MB)")
