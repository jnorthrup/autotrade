"""
Simplified Enhanced Strategies for Testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class SimplifiedEnhancedStrategies:
    """Simplified enhanced trading strategies without scipy/sklearn dependencies"""
    
    def __init__(self):
        pass
        
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        
        return macd, signal_line
    
    def calculate_support_resistance(self, close: pd.Series, lookback: int = 20) -> Dict[str, float]:
        """Identify support and resistance levels"""
        recent_prices = close.tail(lookback)
        
        # Use percentiles to identify support/resistance
        support = np.percentile(recent_prices, 25)
        resistance = np.percentile(recent_prices, 75)
        
        return {
            'support': float(support),
            'resistance': float(resistance),
            'current_price': float(close.iloc[-1]),
            'in_range': support <= close.iloc[-1] <= resistance
        }
    
    def calculate_market_sentiment(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate comprehensive market sentiment"""
        recent_data = df.tail(50)
        
        # Trend analysis
        returns = recent_data['close'].pct_change()
        trend_strength = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Volatility analysis
        volatility = returns.rolling(20).std().iloc[-1]
        
        # RSI sentiment
        rsi = self.calculate_rsi(recent_data['close']).iloc[-1]
        rsi_sentiment = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        
        return {
            'trend_strength': float(trend_strength),
            'volatility': float(volatility),
            'rsi': float(rsi),
            'rsi_sentiment': rsi_sentiment,
            'overall_sentiment': 'BULLISH' if trend_strength > 0.1 else 'BEARISH' if trend_strength < -0.1 else 'NEUTRAL'
        }
    
    def generate_composite_signal(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate composite signal combining multiple strategies"""
        if len(df) < 50:
            return {'signal': 'HOLD', 'strength': 0, 'confidence': 0}
        
        # Get traditional signals
        rsi = self.calculate_rsi(df['close']).iloc[-1]
        macd, signal_line = self.calculate_macd(df['close'])
        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        # Calculate signal weights
        signals = []
        weights = []
        
        # RSI signal
        if rsi < 30:
            signals.append('BUY')
            weights.append(0.3)
        elif rsi > 70:
            signals.append('SELL')
            weights.append(0.3)
        else:
            signals.append('HOLD')
            weights.append(0.2)
        
        # MACD signal
        if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
            signals.append('BUY')
            weights.append(0.3)
        elif current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
            signals.append('SELL')
            weights.append(0.3)
        else:
            signals.append('HOLD')
            weights.append(0.2)
        
        # Bollinger Bands signal
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        current_price = df['close'].iloc[-1]
        
        if current_price <= lower.iloc[-1]:
            signals.append('BUY')
            weights.append(0.2)
        elif current_price >= upper.iloc[-1]:
            signals.append('SELL')
            weights.append(0.2)
        else:
            signals.append('HOLD')
            weights.append(0.1)
        
        # Weighted voting
        signal_votes = {}
        for signal, weight in zip(signals, weights):
            signal_votes[signal] = signal_votes.get(signal, 0) + weight
        
        final_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
        signal_strength = signal_votes[final_signal]
        
        return {
            'signal': final_signal,
            'strength': float(signal_strength * 100),
            'confidence': float(signal_strength),
            'rsi': float(rsi),
            'macd': float(current_macd),
            'signal_line': float(current_signal),
            'market_sentiment': self.calculate_market_sentiment(df)
        }