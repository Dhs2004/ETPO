#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/activate.sh"
cd "$ETPO_ROOT/agent_system/environments/webshop/webshop/search_engine"
# Avoid overwriting an existing shared index.
if [ -e indexes ] || [ -L indexes ]; then
    echo 'An index already exists. Use a fresh checkout to build a new one.' >&2
    exit 1
fi
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py
python -m pyserini.index.lucene --collection JsonCollection --input resources --index indexes --generator DefaultLuceneDocumentGenerator --threads 2 --storePositions --storeDocvectors --storeRaw
