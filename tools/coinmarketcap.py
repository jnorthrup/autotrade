"""
tools/coinmarketcap.py — CoinMarketCap API client.

Uses COINMARKETCAP_API_KEY env var, urllib.request (no new deps),
rate limiting (1 call/3s), cache to ~/.cache/autotrade/cmc/ with TTL.
"""

import os
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autotrade" / "cmc"
DEFAULT_TTL_SECONDS = 3600  # 1 hour
DEFAULT_RATE_LIMIT = 3.0  # seconds between calls


class CMCClient:
    """CoinMarketCap API client with rate limiting and file cache."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.api_key = api_key or os.environ.get("COINMARKETCAP_API_KEY", "")
        self.rate_limit_seconds = rate_limit_seconds
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.ttl_seconds = ttl_seconds
        self._last_call_time: float = 0.0

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_call_time = time.monotonic()

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Generate a cache key from endpoint and params."""
        import hashlib
        param_str = json.dumps(params, sort_keys=True)
        raw = f"{endpoint}:{param_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[dict]:
        """Read from cache if not expired."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            fetched_at = data.get("fetched_at", 0)
            if time.time() - fetched_at > self.ttl_seconds:
                return None  # Expired
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def _cache_write(self, key: str, data: Any):
        """Write data to cache with timestamp."""
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump({"fetched_at": time.time(), "data": data}, f)
        except OSError:
            pass  # Cache write failure is non-fatal

    def _api_call(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an API call with rate limiting and caching."""
        params = params or {}
        cache_key = self._cache_key(endpoint, params)

        # Check cache first
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached["data"]

        # Build URL
        url = f"https://pro-api.coinmarketcap.com{endpoint}"
        query = "&".join(f"{k}={v}" for k, v in params.items())
        if query:
            url += f"?{query}"

        self._rate_limit()

        req = urllib.request.Request(url)
        req.add_header("X-CMC_PRO_API_KEY", self.api_key)
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                # Cache the response
                self._cache_write(cache_key, data)
                return data
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"CMC API error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"CMC connection error: {e.reason}") from e

    def fetch_latest_listings(
        self,
        start: int = 1,
        limit: int = 100,
        convert: str = "USD",
    ) -> List[dict]:
        """Fetch latest cryptocurrency listings from CMC.

        Returns a list of dicts with symbol, rank, price, volume, etc.
        """
        endpoint = "/v1/cryptocurrency/listings/latest"
        params = {"start": start, "limit": limit, "convert": convert}
        response = self._api_call(endpoint, params)
        return self._parse_listings(response)

    def _parse_listings(self, response: dict) -> List[dict]:
        """Parse CMC listings response into flat dicts."""
        results = []
        for coin in response.get("data", []):
            quote = coin.get("quote", {}).get("USD", {})
            results.append({
                "symbol": coin.get("symbol", ""),
                "rank": coin.get("cmc_rank", 0),
                "price_usd": quote.get("price", 0.0),
                "market_cap": quote.get("market_cap", 0.0),
                "volume_24h": quote.get("volume_24h", 0.0),
                "volume_7d": quote.get("volume_7d", 0.0),
                "volume_30d": quote.get("volume_30d", 0.0),
                "percent_change_1h": quote.get("percent_change_1h", 0.0),
                "percent_change_24h": quote.get("percent_change_24h", 0.0),
                "percent_change_7d": quote.get("percent_change_7d", 0.0),
                "percent_change_30d": quote.get("percent_change_30d", 0.0),
                "last_updated": quote.get("last_updated", ""),
            })
        return results

    def fetch_and_store(
        self,
        conn,
        limit: int = 500,
    ):
        """Fetch latest listings and store in DuckDB cmc_rankings table."""
        listings = self.fetch_latest_listings(start=1, limit=limit)
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("DELETE FROM cmc_rankings")
        for item in listings:
            conn.execute(
                """INSERT INTO cmc_rankings
                (symbol, rank, price_usd, market_cap, volume_24h, volume_7d, volume_30d,
                 percent_change_1h, percent_change_24h, percent_change_7d, percent_change_30d,
                 last_updated, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    item["symbol"],
                    item["rank"],
                    item["price_usd"],
                    item["market_cap"],
                    item["volume_24h"],
                    item["volume_7d"],
                    item["volume_30d"],
                    item["percent_change_1h"],
                    item["percent_change_24h"],
                    item["percent_change_7d"],
                    item["percent_change_30d"],
                    item["last_updated"],
                    now,
                ],
            )
        return len(listings)
