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
# Authentication: plain `git push` over HTTPS - git will prompt for
# credentials the first time (username: your HF username, password: a
# write-access token from https://huggingface.co/settings/tokens), and your
# OS credential helper (e.g. macOS Keychain) should cache it after that.
#
# What it does:
#   1. Clones this GitHub repo and the target HF Space repo into a temp dir.
#   2. Copies the repo's code into the Space repo (everything except .git).
#   3. Populates the Space repo's data/ directory, either by extracting the
#      given tarball or by running the ingest pipeline.
#   4. Tracks the binary artifacts (*.parquet, *.gpickle) through git-lfs -
#      Hugging Face rejects a plain `git push` containing untracked binary
#      files, and a prior attempt at this script using the huggingface_hub
#      Python API instead hit a reproducible server-side bug where such
#      commits landed empty. Plain git+lfs is the well-trodden path here.
#   5. Commits and pushes to the Space.
#
# Requires: git, git-lfs, rsync, tar. Uses python3 + a venv only if it has to
# run the ingest pipeline (no cached artifacts available).

set -euo pipefail

GITHUB_REPO_URL="https://github.com/SamV250/fly-connectome-explorer.git"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf-space-url> [data-artifacts-tarball]" >&2
    echo "Example: $0 https://huggingface.co/spaces/<user>/<space-name>" >&2
    exit 1
fi

SPACE_URL="$1"
TARBALL="${2:-./precomputed_data_artifacts.tar.gz}"

for cmd in git git-lfs rsync tar; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' is required but not found." >&2; exit 1; }
done

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Cloning source repo: $GITHUB_REPO_URL"
git clone --depth 1 "$GITHUB_REPO_URL" "$WORKDIR/source"

echo "==> Cloning Space repo: $SPACE_URL"
git clone "$SPACE_URL" "$WORKDIR/space"

echo "==> Copying app code into the Space repo"
rsync -a --exclude='.git' --exclude='.gitignore' --exclude='data/' "$WORKDIR/source/" "$WORKDIR/space/"

# The source repo's .gitignore excludes data/* (by design - it keeps large
# generated artifacts out of GitHub). Copying it verbatim into the Space
# would make `git add` silently skip the very artifacts this script just
# placed in data/, so the Space repo gets its own, permissive one instead.
cat > "$WORKDIR/space/.gitignore" <<'GITIGNORE'
data/raw/
.venv/
__pycache__/
*.pyc
GITIGNORE

mkdir -p "$WORKDIR/space/data"

if [[ -f "$TARBALL" ]]; then
    echo "==> Extracting precomputed artifacts from $TARBALL"
    tar xzf "$TARBALL" -C "$WORKDIR/space/data"
else
    echo "==> No artifact tarball found at $TARBALL; running the ingest pipeline instead"
    echo "    (this downloads ~880 MB from Zenodo and trains node2vec - a few minutes)"

    python3 -m venv "$WORKDIR/venv"
    # shellcheck disable=SC1091
    source "$WORKDIR/venv/bin/activate"
    pip install -q -r "$WORKDIR/source/requirements.txt"

    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/fetch_connectome.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/build_graph.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/compute_graph_stats.py"
    PYTHONPATH="$WORKDIR/source/ingest" python3 "$WORKDIR/source/ingest/compute_embeddings.py"

    cp "$WORKDIR/source/data/graph.gpickle" \
       "$WORKDIR/source/data/graph_stats.parquet" \
       "$WORKDIR/source/data/graph_summary.json" \
       "$WORKDIR/source/data/embeddings.parquet" \
       "$WORKDIR/space/data/"

    deactivate
fi

cp "$WORKDIR/source/data/README.md" "$WORKDIR/space/data/README.md"

for artifact in graph.gpickle graph_stats.parquet graph_summary.json embeddings.parquet; do
    if [[ ! -f "$WORKDIR/space/data/$artifact" ]]; then
        echo "Error: data/$artifact is missing - the Space would fail to start. Aborting before commit." >&2
        exit 1
    fi
done

echo "==> Setting up git-lfs tracking for binary artifacts"
cd "$WORKDIR/space"
git lfs install --local
git lfs track "*.parquet" "*.gpickle"

git add -A

for artifact in graph.gpickle graph_stats.parquet graph_summary.json embeddings.parquet; do
    filter="$(git check-attr filter -- "data/$artifact" | sed 's/.*: filter: //')"
    if [[ "$artifact" == *.parquet || "$artifact" == *.gpickle ]] && [[ "$filter" != "lfs" ]]; then
        echo "Error: data/$artifact is not tracked by git-lfs (filter=$filter) - aborting before commit." >&2
        exit 1
    fi
done

if git diff --cached --quiet; then
    echo "Nothing changed - Space is already up to date."
else
    git commit -m "Deploy Fly Connectome Explorer"
    git push
    echo "==> Pushed. The Space will rebuild automatically at $SPACE_URL"
fi
