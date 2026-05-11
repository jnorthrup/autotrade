
import time
import pandas as pd
import duckdb
from datetime import datetime
import os

# Mocking the part of CandleIngestor we want to benchmark
class MockIngestor:
    def __init__(self, db_path=":memory:"):
        self.db = duckdb.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id VARCHAR PRIMARY KEY,
                base_currency VARCHAR,
                quote_currency VARCHAR,
                status VARCHAR,
                volume_24h DOUBLE
            )
        """)

    def save_products_original(self, products):
        if not products:
            return
        df = pd.DataFrame(products)
        df['last_updated'] = datetime.now()

        # Clear and insert
        self.db.execute("DELETE FROM products")
        for _, row in df.iterrows():
            self.db.execute("""
                INSERT INTO products VALUES (?, ?, ?, ?, ?)
            """, [row['product_id'], row['base_currency'], row['quote_currency'],
                  row['status'], row['volume_24h']])

    def save_products_optimized(self, products):
        if not products:
            return
        df = pd.DataFrame(products)
        # df['last_updated'] = datetime.now() # Not used in original INSERT anyway

        self.db.execute("DELETE FROM products")
        # DuckDB can directly insert from a pandas DataFrame
        self.db.execute("INSERT INTO products SELECT product_id, base_currency, quote_currency, status, volume_24h FROM df")

def run_benchmark(n_products=1000):
    products = [
        {
            'product_id': f'PROD-{i}',
            'base_currency': 'BTC',
            'quote_currency': 'USD',
            'status': 'online',
            'volume_24h': 1234.56 * i
        }
        for i in range(n_products)
    ]

    ingestor = MockIngestor()

    # Benchmark original
    start_time = time.time()
    ingestor.save_products_original(products)
    original_duration = time.time() - start_time
    print(f"Original save_products with {n_products} products: {original_duration:.4f} seconds")

    # Benchmark optimized
    start_time = time.time()
    ingestor.save_products_optimized(products)
    optimized_duration = time.time() - start_time
    print(f"Optimized save_products with {n_products} products: {optimized_duration:.4f} seconds")

    if optimized_duration > 0:
        improvement = (original_duration - optimized_duration) / original_duration * 100
        print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_benchmark(1000)
    run_benchmark(10000)
