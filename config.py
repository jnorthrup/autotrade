import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
    
    # Database
    DB_PATH = Path(os.getenv("DB_PATH", "candles.duckdb")).resolve()
    BAG_PATH = Path(os.getenv("BAG_PATH", "bag.json")).resolve()
    
    # Defaults
    DEFAULT_GRANULARITY = os.getenv("DEFAULT_GRANULARITY", "300")
    
    @classmethod
    def validate(cls):
        """Simple validation to warn if critical keys are missing."""
        if not cls.COINBASE_API_KEY or cls.COINBASE_API_KEY == "your_api_key_here":
            print("WARNING: COINBASE_API_KEY not set or using placeholder.")
        if not cls.COINBASE_API_SECRET or cls.COINBASE_API_SECRET == "your_api_secret_here":
            print("WARNING: COINBASE_API_SECRET not set or using placeholder.")

Config.validate()
