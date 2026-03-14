import json
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any
import requests
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

from config import Config

class CoinbaseAdvancedClient:
    """Enhanced Coinbase Advanced API client for paper trading"""
    
    def __init__(self, sandbox: bool = True):
        self.sandbox = sandbox
        self.base_url = "https://api.exchange.coinbase.com" if not sandbox else "https://api-public.sandbox.exchange.coinbase.com"
        self.api_key = Config.API_KEY
        self.api_secret = Config.API_SECRET
        self.passphrase = Config.API_PASSPHRASE
        
        if not all([self.api_key, self.api_secret, self.passphrase]):
            logger.warning("API credentials not found - using paper trading mode only")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """Generate Coinbase API signature"""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': base64.b64encode(signature).decode(),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make authenticated API request"""
        url = f"{self.base_url}{endpoint}"
        timestamp = str(time.time())
        
        body = json.dumps(data) if data else ""
        headers = self._generate_signature(timestamp, method, endpoint, body)
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if self.sandbox or not all([self.api_key, self.api_secret]):
            return {
                "USD": Config.INITIAL_BALANCE,
                "BTC": 0.0,
                "ETH": 0.0,
                "ADA": 0.0,
                "SOL": 0.0
            }
        
        try:
            accounts = self._make_request('GET', '/accounts')
            balances = {}
            for account in accounts.get('accounts', []):
                currency = account.get('currency')
                balance = float(account.get('available_balance', {}).get('value', 0))
                balances[currency] = balance
            return balances
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}
    
    def get_product_ticker(self, product_id: str) -> Dict[str, Any]:
        """Get current ticker for a product"""
        try:
            if self.sandbox:
                import numpy as np
                base_price = 45000 + np.random.normal(0, 2000)
                return {
                    "trade_id": int(time.time()),
                    "price": f"{base_price:.2f}",
                    "size": "0.1",
                    "time": datetime.now().isoformat(),
                    "bid": f"{base_price - 50:.2f}",
                    "ask": f"{base_price + 50:.2f}",
                    "volume": f"{np.random.uniform(500, 2000):.2f}"
                }
            
            return self._make_request('GET', f'/products/{product_id}/ticker')
        except Exception as e:
            logger.error(f"Failed to get ticker for {product_id}: {e}")
            return {}
    
    def get_product_candles(self, product_id: str, start: str, end: str, granularity: int = 300) -> List[List]:
        """Get historical candle data with realistic price movements"""
        try:
            params = {
                'start': start,
                'end': end,
                'granularity': granularity
            }
            
            if self.sandbox:
                import numpy as np
                base_price = 45000
                num_candles = 200
                candles = []
                start_time = int(time.time()) - (num_candles * granularity)
                
                trend_direction = np.random.choice([-1, 1])
                
                for i in range(num_candles):
                    timestamp = start_time + (i * granularity)
                    
                    trend_component = trend_direction * np.random.uniform(0.005, 0.02)
                    noise = np.random.normal(0, 0.015)
                    
                    open_price = base_price
                    change_pct = trend_component + noise
                    close_price = base_price * (1 + change_pct)
                    
                    high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
                    low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
                    
                    if high_price < low_price:
                        high_price, low_price = low_price, high_price
                    
                    volume = abs(np.random.normal(100, 50))
                    
                    candles.append([timestamp, low_price, high_price, open_price, close_price, volume])
                    base_price = close_price
                    
                    if i > 0 and i % 50 == 0:
                        trend_direction *= -1
                
                return candles
            
            return self._make_request('GET', f'/products/{product_id}/candles', params=params)
        except Exception as e:
            logger.error(f"Failed to get candles for {product_id}: {e}")
            return []
    
    def get_products(self) -> List[Dict[str, Any]]:
        """Get available trading products"""
        try:
            if self.sandbox:
                return [
                    {"id": "BTC-USD", "display_name": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD"},
                    {"id": "ETH-USD", "display_name": "ETH-USD", "base_currency": "ETH", "quote_currency": "USD"},
                    {"id": "ADA-USD", "display_name": "ADA-USD", "base_currency": "ADA", "quote_currency": "USD"},
                    {"id": "SOL-USD", "display_name": "SOL-USD", "base_currency": "SOL", "quote_currency": "USD"}
                ]
            
            return self._make_request('GET', '/products')
        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return []
    
    def get_historical_data(self, product_id: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data as DataFrame"""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        candles = self.get_product_candles(
            product_id,
            start.isoformat(),
            end.isoformat(),
            granularity=3600
        )
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        
        return df