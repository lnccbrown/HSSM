#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  echo "Usage: ./scripts/docs.sh {build|serve}" >&2
}

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

case "$1" in
  build | serve)
    command="$1"
    ;;
  *)
    usage
    exit 2
    ;;
esac

uv sync \
  --python 3.12 \
  --group dev \
  --group notebook \
  --group docs

uv run \
  --python 3.12 \
  --group dev \
  --group notebook \
  --group docs \
  --no-sync \
  python scripts/check_docs_weight.py

uv run \
  --python 3.12 \
  --group dev \
  --group notebook \
  --group docs \
  --no-sync \
  python scripts/check_docs_notebook_paths.py

case "$command" in
  build)
    uv run \
      --python 3.12 \
      --group dev \
      --group notebook \
      --group docs \
      --no-sync \
      mkdocs build --strict
    ;;
  serve)
    uv run \
      --python 3.12 \
      --group dev \
      --group notebook \
      --group docs \
      --no-sync \
      mkdocs serve
    ;;
esac
