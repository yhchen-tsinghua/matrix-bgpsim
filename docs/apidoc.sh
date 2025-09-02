#!/usr/bin/env bash
set -e

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR" || exit 1 

sphinx-apidoc -o source ../matrix_bgpsim
