#!/usr/bin/env bash
# Deploy Fly Connectome Explorer to a Hugging Face Space.
#
# Run this on a machine that can reach huggingface.co (this script is not
# meant to run inside the sandboxed Claude Code environment that built the
# repo, since that environment's network policy blocks huggingface.co).
#
# Usage:
#   ./deploy_to_space.sh <hf-space-url> [data-artifacts-tarball]
#
# Example:
#   ./deploy_to_space.sh https://huggingface.co/spaces/alice/fly-connectome-explorer \
#       ~/Downloads/precomputed_data_artifacts.tar.gz
#
# If you omit the tarball path, the script looks for
# ./precomputed_data_artifacts.tar.gz in the current directory, and if that's
# also missing, it runs the /ingest pipeline from scratch instead (downloads
# ~880 MB from Zenodo and trains node2vec - expect several minutes).
#
# Authentication: set HF_TOKEN to a Hugging Face access token with write
# access (https://huggingface.co/settings/tokens) before running this, e.g.
# `export HF_TOKEN=hf_xxxxx`. If unset, falls back to any token already
# cached by a prior `huggingface-cli login`.
#
# What it does:
#   1. Clones this GitHub repo into a temp dir and assembles the deployable
#      contents (code + data artifacts) in a local folder.
#   2. Uploads that folder to the target Space via the Hugging Face Hub API
#      (huggingface_hub.upload_folder), not raw git - this avoids needing
#      git-lfs/Xet set up locally, which a plain `git push` of the binary
#      data artifacts otherwise gets rejected without.
#
# Requires: git, rsync, tar, python3. Installs huggingface_hub (and, only if
# it has to run the ingest pipeline, this repo's requirements.txt) into a
# throwaway venv - nothing is installed into your regular Python environment.

set -euo pipefail

GITHUB_REPO_URL="https://github.com/SamV250/fly-connectome-explorer.git"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf-space-url> [data-artifacts-tarball]" >&2
    echo "Example: $0 https://huggingface.co/spaces/<user>/<space-name>" >&2
    exit 1
fi

SPACE_URL="$1"
TARBALL="${2:-./precomputed_data_artifacts.tar.gz}"
REPO_ID="${SPACE_URL#https://huggingface.co/spaces/}"
REPO_ID="${REPO_ID%/}"

for cmd in git rsync tar python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' is required but not found." >&2; exit 1; }
done

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Cloning source repo: $GITHUB_REPO_URL"
git clone --depth 1 "$GITHUB_REPO_URL" "$WORKDIR/source"

DEPLOY_DIR="$WORKDIR/deploy"
mkdir -p "$DEPLOY_DIR"
echo "==> Assembling deployable app code"
rsync -a --exclude='.git' --exclude='data/' "$WORKDIR/source/" "$DEPLOY_DIR/"

mkdir -p "$DEPLOY_DIR/data"

python3 -m venv "$WORKDIR/venv"
# shellcheck disable=SC1091
source "$WORKDIR/venv/bin/activate"
pip install -q -U huggingface_hub

if [[ -f "$TARBALL" ]]; then
    echo "==> Extracting precomputed artifacts from $TARBALL"
    tar xzf "$TARBALL" -C "$DEPLOY_DIR/data"
else
    echo "==> No artifact tarball found at $TARBALL; running the ingest pipeline instead"
    echo "    (this downloads ~880 MB from Zenodo and trains node2vec - a few minutes)"

    pip install -q -r "$WORKDIR/source/requirements.txt"

    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/fetch_connectome.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/build_graph.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/compute_graph_stats.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/compute_embeddings.py"

    cp "$WORKDIR/source/data/graph.gpickle" \
       "$WORKDIR/source/data/graph_stats.parquet" \
       "$WORKDIR/source/data/graph_summary.json" \
       "$WORKDIR/source/data/embeddings.parquet" \
       "$DEPLOY_DIR/data/"
fi

cp "$WORKDIR/source/data/README.md" "$DEPLOY_DIR/data/README.md"

for artifact in graph.gpickle graph_stats.parquet graph_summary.json embeddings.parquet; do
    if [[ ! -f "$DEPLOY_DIR/data/$artifact" ]]; then
        echo "Error: data/$artifact is missing - the Space would fail to start. Aborting before upload." >&2
        exit 1
    fi
done

echo "==> Uploading to Space: $REPO_ID"
python3 - "$DEPLOY_DIR" "$REPO_ID" <<'PYEOF'
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

folder_path, repo_id = Path(sys.argv[1]), sys.argv[2]

# Built as explicit per-file "add" operations via create_commit rather than
# upload_folder(): upload_folder diffs the local folder against the repo's
# current state and skips any file it believes is already present, and that
# check can be fooled if a *previous, ultimately-rejected* push already
# transferred the same blob content into the Hub's storage backend (this
# happens with plain `git push` when Hugging Face's pre-receive hook rejects
# the ref update for containing untracked binary files, but only after the
# objects were already received). create_commit has no such shortcut: since
# these paths don't yet exist in the repo's tree, it must add them.
files = [path for path in sorted(folder_path.rglob("*")) if path.is_file()]
operations = [
    CommitOperationAdd(path_in_repo=path.relative_to(folder_path).as_posix(), path_or_fileobj=path.read_bytes())
    for path in files
]

print(f"==> {len(operations)} operations built locally:")
for path, op in zip(files, operations):
    print(f"    {op.path_in_repo}  ({len(op.path_or_fileobj)} bytes)")

api = HfApi()
# A distinct commit message (with the local file count baked in) rules out
# any chance this specific request gets treated as a duplicate of an
# earlier one - some commit-tracking systems dedupe on message+parent, not
# just tree content.
commit_info = api.create_commit(
    repo_id=repo_id,
    repo_type="space",
    operations=operations,
    commit_message=f"Deploy Fly Connectome Explorer ({len(operations)} files)",
)
print(f"==> Committed {commit_info.oid}")

# Verify directly through the API - not a browser page, so it can't be
# stale/cached - that the commit we just made actually put these files
# where the app expects them.
remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="space"))
required = {
    "data/graph.gpickle",
    "data/graph_stats.parquet",
    "data/graph_summary.json",
    "data/embeddings.parquet",
}
missing = required - remote_files
if missing:
    print(f"ERROR: commit {commit_info.oid} landed, but the API still doesn't list: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)

print("==> Verified: all 4 data artifacts are present in the repo per the API (not a cached page).")
print(f"==> The Space will rebuild automatically at https://huggingface.co/spaces/{repo_id}")
PYEOF

deactivate
