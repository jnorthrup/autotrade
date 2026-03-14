import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Enhanced configuration for paper trading bot"""
    
    # Coinbase API Configuration
    API_KEY = os.getenv('COINBASE_API_KEY')
    API_SECRET = os.getenv('COINBASE_API_SECRET')
    API_PASSPHRASE = os.getenv('COINBASE_API_PASSPHRASE')
    SANDBOX = True  # Use sandbox for paper trading
    
    # Paper trading settings
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 100.0))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.95))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', 0.05))
    
    # Trading parameters
    BASE_CURRENCY = 'USD'
    TRADING_PAIRS = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
    PRIMARY_PAIR = 'BTC-USD'
    MIN_ORDER_SIZE = 0.001
    
    # Risk management
    STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.05  # 5% take profit
    MAX_OPEN_POSITIONS = 3
    
    # Strategy parameters
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # File paths
    PAPER_DB_PATH = "paper_trading.db"
    LOG_FILE = "trading_bot.log"
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.PAPER_TRADING and not all([
            cls.API_KEY,
            cls.API_SECRET, 
            cls.API_PASSPHRASE
        ]):
            return False
        return True
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(cls, key))
        }