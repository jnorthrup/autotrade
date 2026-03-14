"""
Multi-Asset Paper Trading with Portfolio Diversification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Asset:
    symbol: str
    name: str
    price: float
    volatility: float
    correlation_matrix: Dict[str, float]  # Correlation with other assets


class MultiAssetPaperTrading:
    """Paper trading with multiple assets and portfolio optimization"""
    
    def __init__(self, initial_balance: float = 100.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # asset_symbol -> {'size': float, 'cost': float, 'entry_price': float}
        self.orders = []
        self.trades = []
        self.assets = self.initialize_assets()
        self.portfolio_history = []
        
    def initialize_assets(self) -> Dict[str, Asset]:
        """Initialize trading assets"""
        assets = {
            'BTC': Asset('BTC', 'Bitcoin', 45000.0, 0.02, {'ETH': 0.7, 'ADA': 0.5, 'SOL': 0.6}),
            'ETH': Asset('ETH', 'Ethereum', 2500.0, 0.025, {'BTC': 0.7, 'ADA': 0.6, 'SOL': 0.7}),
            'ADA': Asset('ADA', 'Cardano', 0.45, 0.03, {'BTC': 0.5, 'ETH': 0.6, 'SOL': 0.8}),
            'SOL': Asset('SOL', 'Solana', 100.0, 0.035, {'BTC': 0.6, 'ETH': 0.7, 'ADA': 0.8}),
        }
        return assets
    
    def update_asset_prices(self):
        """Update asset prices with realistic correlated movements"""
        # Base price movements with correlation
        base_movement = np.random.normal(0, 0.01)
        
        for symbol, asset in self.assets.items():
            # Base movement plus asset-specific noise
            movement = base_movement + np.random.normal(0, asset.volatility * 0.3)
            new_price = asset.price * (1 + movement)
            
            # Ensure price stays positive
            asset.price = max(new_price, asset.price * 0.9)  # Max 10% drop
            
            # Update correlations slightly
            for other_symbol in self.assets:
                if other_symbol != symbol:
                    current_corr = asset.correlation_matrix.get(other_symbol, 0.5)
                    noise = np.random.normal(0, 0.05)
                    asset.correlation_matrix[other_symbol] = max(0.1, min(0.9, current_corr + noise))
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            position['size'] * self.assets[asset].price 
            for asset, position in self.positions.items()
        )
        return self.balance + position_value
    
    def calculate_position_size(self, asset: str, risk_amount: float, 
                              stop_loss_pct: float = 0.02) -> float:
        """Calculate position size for a specific asset"""
        if asset not in self.assets:
            return 0
            
        current_price = self.assets[asset].price
        volatility = self.assets[asset].volatility
        
        # Adjust risk amount based on volatility
        volatility_adjustment = 0.02 / max(volatility, 0.01)  # Normalize to 2% baseline
        adjusted_risk = risk_amount * min(volatility_adjustment, 2.0)  # Cap adjustment
        
        position_value = adjusted_risk / stop_loss_pct
        position_size = position_value / current_price
        
        # Max position size: 25% of portfolio per asset
        max_position = self.get_portfolio_value() * 0.25 / current_price
        
        return min(position_size, max_position)
    
    def buy(self, asset: str, price: float, size: float, order_id: str = None) -> bool:
        """Execute buy order for specific asset"""
        if size <= 0:
            return False
            
        cost = price * size
        if cost > self.balance:
            logger.warning(f"Insufficient funds for {asset}. Required: ${cost:.2f}, Available: ${self.balance:.2f}")
            return False
        
        # Update balance
        self.balance -= cost
        
        # Update positions
        if asset in self.positions:
            self.positions[asset]['size'] += size
            self.positions[asset]['cost'] += cost
            self.positions[asset]['entry_price'] = (self.positions[asset]['cost'] / self.positions[asset]['size'])
        else:
            self.positions[asset] = {
                'size': size,
                'cost': cost,
                'entry_price': price
            }
        
        # Record order
        order = {
            'order_id': order_id or f"buy_{asset}_{len(self.orders)}",
            'asset': asset,
            'side': 'buy',
            'size': size,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now(),
            'status': 'filled'
        }
        self.orders.append(order)
        self.trades.append(order.copy())  # Copy as trade
        
        logger.info(f"BUY {asset}: {size:.6f} @ ${price:.2f} = ${cost:.2f}")
        return True
    
    def sell(self, asset: str, price: float, size: float, order_id: str = None) -> bool:
        """Execute sell order for specific asset"""
        if asset not in self.positions or self.positions[asset]['size'] < size:
            logger.warning(f"Insufficient {asset} to sell")
            return False
        
        proceeds = price * size
        
        # Update balance
        self.balance += proceeds
        
        # Update positions
        self.positions[asset]['size'] -= size
        cost_basis = (self.positions[asset]['cost'] / 
                     (self.positions[asset]['size'] + size)) * size
        self.positions[asset]['cost'] -= cost_basis
        
        # Calculate profit/loss
        profit = proceeds - cost_basis
        
        if self.positions[asset]['size'] <= 0:
            del self.positions[asset]
        
        # Record trade
        trade = {
            'order_id': order_id or f"sell_{asset}_{len(self.trades)}",
            'asset': asset,
            'side': 'sell',
            'size': size,
            'price': price,
            'proceeds': proceeds,
            'cost_basis': cost_basis,
            'profit': profit,
            'timestamp': datetime.now(),
            'status': 'filled'
        }
        self.orders.append(trade)
        self.trades.append(trade)
        
        logger.info(f"SELL {asset}: {size:.6f} @ ${price:.2f} = ${proceeds:.2f} (Profit: ${profit:.2f})")
        return True
    
    def calculate_portfolio_metrics(self) -> Dict[str, any]:
        """Calculate comprehensive portfolio metrics"""
        portfolio_value = self.get_portfolio_value()
        current_prices = {asset: self.assets[asset].price for asset in self.assets}
        
        # Position values
        position_values = {}
        for asset, position in self.positions.items():
            if asset in current_prices:
                position_values[asset] = position['size'] * current_prices[asset]
        
        # Portfolio weights
        total_position_value = sum(position_values.values())
        weights = {}
        if total_position_value > 0:
            weights = {asset: value/total_position_value for asset, value in position_values.items()}
        
        # Calculate diversification metrics
        num_assets = len(self.positions)
        herfindahl_index = sum(w**2 for w in weights.values()) if weights else 0
        diversification_score = 1 - herfindahl_index  # Higher = more diversified
        
        # Calculate correlation-adjusted risk
        correlation_risk = self.calculate_correlation_risk(weights)
        
        # Performance metrics
        total_return = portfolio_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        metrics = {
            'portfolio_value': portfolio_value,
            'cash_balance': self.balance,
            'position_values': position_values,
            'portfolio_weights': weights,
            'num_assets': num_assets,
            'diversification_score': diversification_score,
            'herfindahl_index': herfindahl_index,
            'correlation_risk': correlation_risk,
            'total_return': total_return,
            'return_pct': return_pct,
            'current_prices': current_prices,
            'timestamp': datetime.now()
        }
        
        # Store in history
        self.portfolio_history.append(metrics)
        
        return metrics
    
    def calculate_correlation_risk(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio correlation risk"""
        if len(weights) <= 1:
            return 0
        
        # Get correlation matrix from first asset (they all have the same matrix)
        asset = list(weights.keys())[0]
        if asset in self.assets:
            correlation_matrix = self.assets[asset].correlation_matrix
        else:
            return 0
        
        # Calculate weighted correlation
        weighted_correlation = 0
        assets = list(weights.keys())
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j:
                    corr = correlation_matrix.get(asset2, 0.5)
                    weighted_correlation += weights[asset1] * weights[asset2] * corr
        
        # Normalize
        correlation_risk = weighted_correlation / (len(assets) * (len(assets) - 1))
        
        return correlation_risk
    
    def optimize_portfolio_allocation(self, risk_tolerance: float = 0.5) -> Dict[str, float]:
        """Optimize portfolio allocation based on risk tolerance"""
        # Calculate expected returns (simplified: inverse volatility)
        expected_returns = {}
        for symbol, asset in self.assets.items():
            # Higher volatility = higher potential return (risk/return tradeoff)
            expected_returns[symbol] = asset.volatility * 100  # Scale up
        
        # Calculate optimal weights using Markowitz-like optimization (simplified)
        total_expected_return = sum(expected_returns.values())
        if total_expected_return == 0:
            return {}
        
        # Base weights: proportional to expected return
        base_weights = {symbol: ret/total_expected_return for symbol, ret in expected_returns.items()}
        
        # Adjust for risk tolerance
        # Higher risk tolerance = more concentrated in high-return assets
        adjusted_weights = {}
        for symbol, weight in base_weights.items():
            asset_risk = self.assets[symbol].volatility
            
            if risk_tolerance > 0.7:  # High risk tolerance
                adjustment = 1 + (asset_risk * 2)
            elif risk_tolerance < 0.3:  # Low risk tolerance
                adjustment = 1 / (1 + asset_risk)
            else:  # Medium risk tolerance
                adjustment = 1
            
            adjusted_weights[symbol] = weight * adjustment
        
        # Normalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Ensure no position exceeds 40%
        final_weights = {}
        for symbol, weight in adjusted_weights.items():
            final_weights[symbol] = min(weight, 0.4)
        
        # Renormalize
        total_final = sum(final_weights.values())
        if total_final > 0:
            final_weights = {k: v/total_final for k, v in final_weights.items()}
        
        return final_weights
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           rebalance_threshold: float = 0.05):
        """Rebalance portfolio to target weights"""
        current_metrics = self.calculate_portfolio_metrics()
        portfolio_value = current_metrics['portfolio_value']
        current_weights = current_metrics['portfolio_weights']
        
        # Initialize current weights for all assets
        for asset in target_weights:
            if asset not in current_weights:
                current_weights[asset] = 0
        
        # Check if rebalancing is needed
        rebalance_needed = False
        for asset, target_weight in target_weights.items():
            current_weight = current_weights.get(asset, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > rebalance_threshold:
                rebalance_needed = True
                break
        
        if not rebalance_needed:
            logger.info("Portfolio within target weights, no rebalance needed")
            return
        
        logger.info("Rebalancing portfolio...")
        
        # Calculate target values
        target_values = {}
        for asset, weight in target_weights.items():
            target_values[asset] = portfolio_value * weight
        
        # Calculate current values
        current_values = {}
        for asset in target_weights:
            if asset in self.positions:
                current_values[asset] = self.positions[asset]['size'] * self.assets[asset].price
            else:
                current_values[asset] = 0
        
        # Calculate trades needed
        for asset in target_weights:
            target_value = target_values[asset]
            current_value = current_values[asset]
            trade_value = target_value - current_value
            
            if abs(trade_value) > portfolio_value * 0.01:  # Minimum trade 1%
                if trade_value > 0:
                    # Buy
                    size = trade_value / self.assets[asset].price
                    self.buy(asset, self.assets[asset].price, size, f"rebalance_buy_{asset}")
                else:
                    # Sell
                    size = -trade_value / self.assets[asset].price
                    if asset in self.positions:
                        size = min(size, self.positions[asset]['size'])
                        self.sell(asset, self.assets[asset].price, size, f"rebalance_sell_{asset}")
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get comprehensive performance summary"""
        metrics = self.calculate_portfolio_metrics()
        
        # Calculate trade metrics
        buy_trades = [t for t in self.trades if t['side'] == 'buy']
        sell_trades = [t for t in self.trades if t['side'] == 'sell']
        
        total_buys = len(buy_trades)
        total_sells = len(sell_trades)
        
        # Calculate profit metrics from sell trades
        sell_profits = [t['profit'] for t in sell_trades]
        winning_trades = [p for p in sell_profits if p > 0]
        
        summary = {
            'initial_balance': self.initial_balance,
            'current_balance': metrics['portfolio_value'],
            'total_return': metrics['total_return'],
            'return_pct': metrics['return_pct'],
            'cash_balance': self.balance,
            'num_positions': len(self.positions),
            'positions': {asset: {'size': pos['size'], 'value': pos['size'] * self.assets[asset].price}
                         for asset, pos in self.positions.items()},
            'diversification_score': metrics['diversification_score'],
            'correlation_risk': metrics['correlation_risk'],
            'total_trades': len(self.trades),
            'total_buys': total_buys,
            'total_sells': total_sells,
            'win_rate': len(winning_trades) / len(sell_profits) if sell_profits else 0,
            'avg_profit': np.mean(sell_profits) if sell_profits else 0,
            'total_profit': sum(sell_profits),
            'max_profit': max(sell_profits) if sell_profits else 0,
            'max_loss': min(sell_profits) if sell_profits else 0,
            'timestamp': datetime.now()
        }
        
        return summary
    
    def save_portfolio_state(self, filepath: str = "portfolio_state.json"):
        """Save current portfolio state to file"""
        state = {
            'balance': self.balance,
            'positions': self.positions,
            'assets': {symbol: {'price': asset.price, 'volatility': asset.volatility}
                      for symbol, asset in self.assets.items()},
            'portfolio_history': [{
                'timestamp': m['timestamp'].isoformat() if isinstance(m['timestamp'], datetime) else m['timestamp'],
                'portfolio_value': m['portfolio_value']
            } for m in self.portfolio_history[-100:]],  # Keep last 100 records
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Portfolio state saved to {filepath}")
    
    def load_portfolio_state(self, filepath: str = "portfolio_state.json"):
        """Load portfolio state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.balance = state['balance']
            self.positions = state['positions']
            
            # Update asset prices
            for symbol, asset_data in state['assets'].items():
                if symbol in self.assets:
                    self.assets[symbol].price = asset_data['price']
                    self.assets[symbol].volatility = asset_data['volatility']
            
            logger.info(f"Portfolio state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            return False