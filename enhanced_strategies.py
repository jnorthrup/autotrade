"""
Enhanced Trading Strategies with ML Signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedTradingStrategies:
    """Advanced trading strategies with ML integration"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.ml_model = None
        
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility measurement"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_vwap(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        return (close * volume).cumsum() / volume.cumsum()
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
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
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML prediction"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift())
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_sma'] = df['volume'].rolling(10).mean()
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['close'])
        features['macd'], features['signal'] = self.calculate_macd(df['close'])
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # Statistical features
        features['skewness'] = df['close'].rolling(20).apply(lambda x: stats.skew(x))
        features['kurtosis'] = df['close'].rolling(20).apply(lambda x: stats.kurtosis(x))
        
        # Target variable: future return (next period)
        features['target'] = df['close'].shift(-1).pct_change()
        features['target_binary'] = (features['target'] > 0).astype(int)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
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
    
    def train_ml_model(self, features: pd.DataFrame):
        """Train ML model for price prediction"""
        # Prepare data
        X = features.drop(['target', 'target_binary'], axis=1).values
        y = features['target_binary'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.ml_model.score(X_train, y_train)
        test_score = self.ml_model.score(X_test, y_test)
        
        logger.info(f"ML Model trained - Train Accuracy: {train_score:.2%}, Test Accuracy: {test_score:.2%}")
        
        return test_score
    
    def generate_ml_signal(self, current_features: np.ndarray) -> Dict[str, any]:
        """Generate trading signal using ML model"""
        if self.ml_model is None:
            return {'signal': 'HOLD', 'confidence': 0, 'ml_model': False}
        
        try:
            prediction = self.ml_model.predict_proba(current_features.reshape(1, -1))
            confidence = float(prediction[0].max())
            predicted_class = int(prediction[0].argmax())
            
            if predicted_class == 1 and confidence > 0.6:
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'ml_model': True,
                    'prediction': 'UP',
                    'strength': confidence * 100
                }
            elif predicted_class == 0 and confidence > 0.6:
                return {
                    'signal': 'SELL',
                    'confidence': confidence,
                    'ml_model': True,
                    'prediction': 'DOWN',
                    'strength': confidence * 100
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': confidence,
                    'ml_model': True,
                    'prediction': 'NEUTRAL',
                    'strength': abs(confidence - 0.5) * 100
                }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'ml_model': False}
    
    def calculate_market_sentiment(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate comprehensive market sentiment"""
        recent_data = df.tail(50)
        
        # Trend analysis
        returns = recent_data['close'].pct_change()
        trend_strength = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Volatility analysis
        volatility = returns.rolling(20).std().iloc[-1]
        volatility_ratio = volatility / returns.abs().mean() if returns.abs().mean() > 0 else 0
        
        # Volume analysis
        volume_trend = recent_data['volume'].pct_change().mean()
        
        # RSI sentiment
        rsi = self.calculate_rsi(recent_data['close']).iloc[-1]
        rsi_sentiment = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        
        return {
            'trend_strength': float(trend_strength),
            'volatility': float(volatility),
            'volatility_ratio': float(volatility_ratio),
            'volume_trend': float(volume_trend),
            'rsi': float(rsi),
            'rsi_sentiment': rsi_sentiment,
            'overall_sentiment': 'BULLISH' if trend_strength > 0.1 else 'BEARISH' if trend_strength < -0.1 else 'NEUTRAL'
        }
    
    def generate_composite_signal(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate composite signal combining multiple strategies"""
        # Prepare ML features
        features = self.prepare_ml_features(df)
        
        if len(features) > 0:
            # Train model if not trained
            if self.ml_model is None and len(features) > 100:
                self.train_ml_model(features)
            
            # Get ML signal
            current_features = features.drop(['target', 'target_binary'], axis=1).iloc[-1].values
            ml_signal = self.generate_ml_signal(current_features)
            
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
            
            # ML signal
            if ml_signal['ml_model']:
                signals.append(ml_signal['signal'])
                weights.append(0.4)
            
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
                'ml_confidence': ml_signal.get('confidence', 0),
                'rsi': float(rsi),
                'macd': float(current_macd),
                'signal_line': float(current_signal),
                'ml_active': ml_signal['ml_model'],
                'market_sentiment': self.calculate_market_sentiment(df)
            }
        
        return {'signal': 'HOLD', 'strength': 0, 'confidence': 0, 'ml_active': False}


class PortfolioAllocator:
    """Portfolio allocation with risk parity"""
    
    def __init__(self, assets: List[str], total_capital: float = 100.0):
        self.assets = assets
        self.total_capital = total_capital
        self.allocations = {}
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() > 0:
            return float(excess_returns.mean() / returns.std() * np.sqrt(252))
        return 0
    
    def optimize_portfolio(self, asset_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Optimize portfolio using risk parity"""
        if not asset_returns:
            return {}
        
        # Calculate metrics for each asset
        asset_metrics = {}
        for asset, returns in asset_returns.items():
            if len(returns) > 0:
                volatility = returns.std()
                sharpe = self.calculate_sharpe_ratio(returns)
                var = self.calculate_var(returns)
                
                asset_metrics[asset] = {
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'var': var,
                    'score': sharpe / max(volatility, 0.0001)  # Avoid division by zero
                }
        
        if not asset_metrics:
            return {}
        
        # Sort by score
        sorted_assets = sorted(asset_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Allocate capital based on scores (risk parity)
        total_score = sum(metric['score'] for _, metric in sorted_assets)
        
        allocations = {}
        for asset, metrics in sorted_assets:
            allocation = (metrics['score'] / total_score) * self.total_capital
            allocations[asset] = max(min(allocation, self.total_capital * 0.5), 0)  # Max 50% per asset
        
        return allocations
    
    def calculate_portfolio_metrics(self, allocations: Dict[str, float], 
                                   asset_prices: Dict[str, float]) -> Dict[str, any]:
        """Calculate portfolio metrics"""
        portfolio_value = sum(allocations.get(asset, 0) for asset in self.assets)
        
        return {
            'total_value': portfolio_value,
            'allocations': allocations,
            'diversification_score': len([a for a in allocations.values() if a > 0]) / len(self.assets),
            'risk_score': sum(allocations.values()) / (portfolio_value + 0.0001),
            'num_positions': len([a for a in allocations.values() if a > 0])
        }