#!/usr/bin/env python3
import duckdb
import pandas as pd
import time
import argparse

def tail_results(limit=10):
    try:
        # Use read_only to avoid lock contention with the active training loop
        with duckdb.connect('candles.duckdb', read_only=True) as conn:
            df = conn.execute(f'''
                SELECT 
                    timestamp, 
                    ROUND(val_bpb, 6) as val_bpb, 
                    params 
                FROM experiments 
                ORDER BY val_bpb ASC 
                LIMIT {limit}
            ''').df()
            
            if df.empty:
                print("No experiments committed yet. Wait for the first epoch to finish.")
                return
                
            print(f"\n=== Top {limit} Best Configurations (by lowest validation bits-per-bar) ===")
            pd.set_option('display.max_colwidth', None)
            print(df.to_string(index=False))
            print("=======================================================================\n")
            
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch autoresearch results")
    parser.add_argument("--watch", action="store_true", help="Continuously poll for new results")
    parser.add_argument("--limit", type=int, default=10, help="Number of top results to show")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                tail_results(args.limit)
                print("Monitoring... (Ctrl+C to stop)")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        tail_results(args.limit)
