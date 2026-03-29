#!/bin/bash
set -e

echo "=========================================="
echo "HRM Trading System - Container Entry"
echo "=========================================="

# Validate environment
if [[ -z "$COINBASE_API_KEY" || -z "$COINBASE_API_SECRET" ]]; then
    echo "WARNING: COINBASE_API_KEY or COINBASE_API_SECRET not set"
    echo "Paper trading may fail if using live data"
fi

# Ensure data directories exist
mkdir -p /app/data /app/checkpoints /app/logs

echo "Environment:"
echo "  DB_PATH=$DB_PATH"
echo "  BAG_PATH=$BAG_PATH"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "  LOG_DIR=$LOG_DIR"

echo ""
echo "Starting: $@"
echo "=========================================="

# Run the command passed to the container
exec "$@"
