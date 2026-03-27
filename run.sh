#!/bin/bash
set -e

cd "$(dirname "$0")"

# Build once with SwiftPM's incremental compiler, then run the built binary directly.
echo "Building Autotrade incrementally..."
swift build --product Autotrade

BIN_DIR="$(swift build --show-bin-path)"
BIN_PATH="${BIN_DIR}/Autotrade"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Autotrade binary not found at ${BIN_PATH}" >&2
  exit 1
fi

# Run with arguments passed to this script
echo "Running Autotrade..."
"${BIN_PATH}" "$@"

echo "Run complete: $(date)"
