#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl --fail --location --output "$work/sqlite.zip" https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip
unzip -q "$work/sqlite.zip" -d "$work"
mkdir -p vendor
cp "$work/sqlite-amalgamation-3450300/"* vendor/
