e#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIKESHED_DIR="${SCRIPT_DIR}/../mp/money/TrikeShed"
OUTPUT_DIR="${SCRIPT_DIR}/build/presentation/Dreamer-1.999"

stamp_package_version() {
  local package_json="$1"
  if [[ -f "${package_json}" ]]; then
    perl -0pi -e 's/"version":\s*"[^"]+"/"version": "1.999"/' "${package_json}"
  fi
}

if [[ ! -x "${TRIKESHED_DIR}/gradlew" ]]; then
  echo "error: expected Gradle wrapper at ${TRIKESHED_DIR}/gradlew" >&2
  exit 1
fi

echo "==> Building Dreamer 1.999 presentation bundle"
echo "    Dreamer project : ${SCRIPT_DIR}"
echo "    TrikeShed root  : ${TRIKESHED_DIR}"

"${TRIKESHED_DIR}/gradlew" -p "${SCRIPT_DIR}" --rerun-tasks jvmTest stageDreamer1999Presentation

stamp_package_version "${SCRIPT_DIR}/build/standalone/dreamer-json-tool/package.json"
stamp_package_version "${OUTPUT_DIR}/wasm/package.json"

echo "==> Dreamer 1.999 bundle ready"
echo "    Output: ${OUTPUT_DIR}"
echo "    Includes: Dreamer 1.999.js + wasm/ TrikeShed-linked standalone outputs"