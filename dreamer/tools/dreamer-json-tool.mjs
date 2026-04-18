#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { dreamerJsonParse, dreamerJsonQuery, dreamerJsonQueryType } from './kotlin/dreamer-json-support.mjs';

function usage() {
  console.error('Usage: node dreamer-json-tool.mjs (--file <path> | --json <json>) [--query <path>] [--type]');
}

function parseArgs(argv) {
  const options = { query: '' };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--file' || arg === '-f') {
      options.file = argv[++i];
    } else if (arg === '--json' || arg === '-j') {
      options.json = argv[++i];
    } else if (arg === '--query' || arg === '-q') {
      options.query = argv[++i] ?? '';
    } else if (arg === '--type' || arg === '-t') {
      options.typeOnly = true;
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

async function readInput(options) {
  if (typeof options.json === 'string') {
    return options.json;
  }
  if (typeof options.file === 'string') {
    return readFile(options.file, 'utf8');
  }
  throw new Error('Provide either --file <path> or --json <json>.');
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    usage();
    return;
  }

  const text = await readInput(options);
  const query = options.query ?? '';
  const output = options.typeOnly
    ? dreamerJsonQueryType(text, query)
    : query.length > 0
      ? dreamerJsonQuery(text, query)
      : dreamerJsonParse(text);

  process.stdout.write(`${output}\n`);
}

main().catch((error) => {
  console.error(error.message);
  usage();
  process.exitCode = 1;
});