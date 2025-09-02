#!/usr/bin/env bash
set -e

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR" || exit 1 

sphinx-build -b html source build/html
