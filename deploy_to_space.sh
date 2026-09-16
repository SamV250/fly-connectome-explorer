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
from huggingface_hub import HfApi

folder_path, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="space",
    commit_message="Deploy Fly Connectome Explorer",
)
print(f"==> Uploaded. The Space will rebuild automatically at https://huggingface.co/spaces/{repo_id}")
PYEOF

deactivate
