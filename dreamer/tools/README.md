# Dreamer 1.999 standalone Wasm presentation

This directory contains the wrapper assets for the Dreamer/TrikeShed Wasm presentation bundle.

- Build the versioned bundle with `bash ./build-dreamer-1.999.sh` from the `dreamer/` folder.
- The versioned presentation lands in `build/presentation/Dreamer-1.999/`.
- That bundle contains:
  - `Dreamer 1.999.js` — the versioned Dreamer shell
  - `wasm/` — the standalone Dreamer Wasm JSON tool plus vendored Kotlin/TrikeShed runtime outputs
- Run the bundled Wasm tool with `node build/presentation/Dreamer-1.999/wasm/dreamer-json-tool.mjs --file sample.json --query bodies/0/name`.
- The bundled Wasm package vendors the generated Kotlin/Wasm runtime plus the resolved `@js-joda/core` runtime dependency, so no extra install step is required inside the staged folder.

The underlying Wasm module is produced by Dreamer and statically links against the TrikeShed Kotlin MPP Wasm library.
