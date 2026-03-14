import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, paper_trading):
        self.paper_trading = paper_trading
        self.signals = []
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate buy/sell signals based on strategy"""
        raise NotImplementedError("Subclasses must implement generate_signals")

class MovingAverageCrossover(TradingStrategy):
    def __init__(self, paper_trading, short_window=20, long_window=50):
        super().__init__(paper_trading)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate signals based on moving average crossover"""
        if len(data) < self.long_window:
            return {'signal': 'HOLD', 'strength': 0}
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        current_price = data['close'].iloc[-1]
        short_ma = data['short_ma'].iloc[-1]
        long_ma = data['long_ma'].iloc[-1]
        
        # Calculate signal strength
        ma_diff = (short_ma - long_ma) / long_ma * 100
        
        # Buy signal: short MA crosses above long MA
        if short_ma > long_ma and data['short_ma'].iloc[-2] <= data['long_ma'].iloc[-2]:
            return {'signal': 'BUY', 'strength': abs(ma_diff)}
        
        # Sell signal: short MA crosses below long MA
        elif short_ma < long_ma and data['short_ma'].iloc[-2] >= data['long_ma'].iloc[-2]:
            return {'signal': 'SELL', 'strength': abs(ma_diff)}
        
        return {'signal': 'HOLD', 'strength': abs(ma_diff)}

class RSIStrategy(TradingStrategy):
    def __init__(self, paper_trading, rsi_period=14, oversold=30, overbought=70):
        super().__init__(paper_trading)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate signals based on RSI"""
        if len(data) < self.rsi_period + 1:
            return {'signal': 'HOLD', 'strength': 0}
        
        rsi = self.calculate_rsi(data['close'])
        
        # Buy signal: RSI oversold
        if rsi < self.oversold:
            strength = (self.oversold - rsi) / self.oversold
            return {'signal': 'BUY', 'strength': strength * 100}
        
        # Sell signal: RSI overbought
        elif rsi > self.overbought:
            strength = (rsi - self.overbought) / (100 - self.overbought)
            return {'signal': 'SELL', 'strength': strength * 100}
        
        return {'signal': 'HOLD', 'strength': abs(rsi - 50)}

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, paper_trading, period=20, std_dev=2):
        super().__init__(paper_trading)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate signals based on Bollinger Bands"""
        if len(data) < self.period:
            return {'signal': 'HOLD', 'strength': 0}
        
        # Calculate Bollinger Bands
        data['middle_band'] = data['close'].rolling(window=self.period).mean()
        data['std'] = data['close'].rolling(window=self.period).std()
        data['upper_band'] = data['middle_band'] + (data['std'] * self.std_dev)
        data['lower_band'] = data['middle_band'] - (data['std'] * self.std_dev)
        
        current_price = data['close'].iloc[-1]
        upper_band = data['upper_band'].iloc[-1]
        lower_band = data['lower_band'].iloc[-1]
        
        # Calculate position in bands (0-100 scale)
        band_range = upper_band - lower_band
        if band_range > 0:
            position = ((current_price - lower_band) / band_range) * 100
        else:
            position = 50
        
        # Buy signal: price at lower band
        if current_price <= lower_band:
            strength = abs(lower_band - current_price) / lower_band * 100
            return {'signal': 'BUY', 'strength': strength}
        
        # Sell signal: price at upper band
        elif current_price >= upper_band:
            strength = abs(current_price - upper_band) / upper_band * 100
            return {'signal': 'SELL', 'strength': strength}
        
        return {'signal': 'HOLD', 'strength': abs(position - 50)}

class MACDStrategy(TradingStrategy):
    def __init__(self, paper_trading, fast=12, slow=26, signal=9):
        super().__init__(paper_trading)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate signals based on MACD"""
        if len(data) < self.slow + self.signal:
            return {'signal': 'HOLD', 'strength': 0}
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.fast).mean()
        exp2 = data['close'].ewm(span=self.slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=self.signal).mean()
        
        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        # Buy signal: MACD crosses above signal
        if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
            strength = abs(current_macd - current_signal)
            return {'signal': 'BUY', 'strength': strength}
        
        # Sell signal: MACD crosses below signal
        elif current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
            strength = abs(current_signal - current_macd)
            return {'signal': 'SELL', 'strength': strength}
        
        return {'signal': 'HOLD', 'strength': abs(current_macd - current_signal)}