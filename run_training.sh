#!/bin/bash
# run_training.sh - Launch HRM training loops in byobu with literbike pool
# Usage: ./run_training.sh        (start all)
#        ./run_training.sh status  (show status)
#        ./run_training.sh kill    (kill all)
#        ./run_training.sh restart (kill + start)

set -e
cd "$(dirname "$0")"

SESSION="hrm-training"
POOL_BIN="./literbike-pool/target/release/duckdb_pool"
DB_PATH="candles.duckdb"
POOL_SOCKET="/tmp/duckdb_pool.sock"

case "${1:-start}" in
  start)
    # Kill existing session if any
    byobu kill-session -t "$SESSION" 2>/dev/null || true
    rm -f "$POOL_SOCKET"

    # Create new detached session with pool window first
    byobu new-session -d -s "$SESSION" -n "pool"

    # Window 0: literbike pool server (must start first, all others depend on it)
    byobu send-keys -t "$SESSION:pool" \
      "$POOL_BIN $DB_PATH --socket $POOL_SOCKET 2>&1 | tee -a logs/pool.log" Enter

    # Wait for pool socket to appear
    echo -n "Waiting for pool server..."
    for i in $(seq 1 30); do
      if [ -S "$POOL_SOCKET" ]; then
        echo " ready."
        break
      fi
      sleep 0.5
    done
    if [ ! -S "$POOL_SOCKET" ]; then
      echo " FAILED - pool socket never appeared"
      exit 1
    fi

    # Window 1: pretrain worker (stochastic bags, binance bulk data)
    byobu new-window -t "$SESSION" -n "pretrain"
    byobu send-keys -t "$SESSION:pretrain" \
      "python3 training_worker.py --mode pretrain --worker-id pretrain-1 --device mps" Enter

    # Window 2: finetune worker (fixed bag, coinbase live data)
    byobu new-window -t "$SESSION" -n "finetune"
    byobu send-keys -t "$SESSION:finetune" \
      "python3 training_worker.py --mode finetune --lookback-days 60 --device mps" Enter

    # Window 3: graph_showdown autoresearch (continuous model proving)
    byobu new-window -t "$SESSION" -n "showdown"
    byobu send-keys -t "$SESSION:showdown" \
      "python3 graph_showdown.py --autoresearch --checkpoint-path model_weights_best.pt --device mps --save-checkpoint" Enter

    # Window 4: standalone finetune cycle (periodic, lighter)
    byobu new-window -t "$SESSION" -n "daytrade"
    byobu send-keys -t "$SESSION:daytrade" \
      "while true; do python3 finetune.py --pretrained model_weights.pt --bag bag.json --lr 0.0001 --exchange coinbase --device mps --output model_weights_daytrade.pt; echo '[daytrade] cycle done, sleeping 30m'; sleep 1800; done" Enter

    echo "Started byobu session '$SESSION' with 5 windows:"
    echo "  0:pool       - literbike duckdb_pool (singleton connection server)"
    echo "  1:pretrain   - training_worker --mode pretrain (stochastic bags)"
    echo "  2:finetune   - training_worker --mode finetune (fixed bag, coinbase)"
    echo "  3:showdown   - graph_showdown --autoresearch (model proving)"
    echo "  4:daytrade   - finetune.py loop (periodic 30m cycle)"
    echo ""
    echo "All Python processes route DuckDB through pool at $POOL_SOCKET"
    echo "Attach: byobu attach -t $SESSION"
    ;;

  status)
    echo "=== Byobu ==="
    byobu list-sessions 2>/dev/null || echo "No byobu sessions"
    echo ""
    echo "=== Windows ==="
    byobu list-windows -t "$SESSION" 2>/dev/null || echo "No '$SESSION' session"
    echo ""
    echo "=== Pool socket ==="
    if [ -S "$POOL_SOCKET" ]; then
      echo "  $POOL_SOCKET EXISTS"
    else
      echo "  $POOL_SOCKET MISSING"
    fi
    echo ""
    echo "=== Python processes ==="
    ps aux | grep -E "training_worker|graph_showdown|finetune\.py|duckdb_pool" | grep -v grep || echo "  none"
    ;;

  kill)
    byobu kill-session -t "$SESSION" 2>/dev/null && echo "Killed '$SESSION'" || echo "No '$SESSION' session found"
    rm -f "$POOL_SOCKET"
    ;;

  restart)
    "$0" kill
    sleep 2
    "$0" start
    ;;

  *)
    echo "Usage: $0 {start|status|kill|restart}"
    exit 1
    ;;
esac
