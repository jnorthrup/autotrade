"""
Enhanced Risk Management with Trailing Stops
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AdvancedRiskManager:
    """Advanced risk management with trailing stops and portfolio optimization"""
    
    def __init__(self, initial_capital: float = 100.0):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.trailing_stops = {}
        self.position_sizes = {}
        self.max_drawdown = 0.0
        self.drawdown_history = []
        
    def calculate_position_size_kelly(self, win_rate: float, avg_win: float, avg_loss: float, 
                                     max_kelly_fraction: float = 0.25) -> float:
        """Calculate position size using Kelly Criterion"""
        if avg_loss == 0 or avg_win == 0:
            return 0
            
        # Kelly fraction: f* = p/a - q/b where p=win prob, q=loss prob, a=avg loss, b=avg win
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p
        
        if b == 0:
            return 0
            
        kelly_fraction = p - (q / b)
        
        # Cap at maximum fraction to avoid excessive risk
        capped_fraction = min(kelly_fraction, max_kelly_fraction)
        
        return max(capped_fraction, 0)
    
    def calculate_var_position_size(self, portfolio_value: float, volatility: float, 
                                   confidence_level: float = 0.95, max_risk: float = 0.02) -> float:
        """Calculate position size based on Value at Risk"""
        if volatility <= 0:
            return 0
            
        # Simplified VaR calculation: portfolio_value * z_score * volatility
        z_score = abs(np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100))
        var_amount = portfolio_value * z_score * volatility
        
        # Risk budget allocation
        risk_budget = portfolio_value * max_risk
        position_value = risk_budget / (z_score * volatility)
        
        return min(position_value, portfolio_value * 0.5)  # Max 50% of portfolio
    
    def set_trailing_stop(self, asset: str, entry_price: float, stop_percent: float = 0.02,
                         trail_percent: float = 0.01):
        """Set trailing stop for a position"""
        initial_stop = entry_price * (1 - stop_percent)
        
        self.trailing_stops[asset] = {
            'entry_price': entry_price,
            'initial_stop': initial_stop,
            'current_stop': initial_stop,
            'highest_price': entry_price,
            'trail_percent': trail_percent,
            'active': True
        }
        
        logger.info(f"Trailing stop set for {asset}: Entry=${entry_price:.2f}, Initial Stop=${initial_stop:.2f}")
    
    def update_trailing_stop(self, asset: str, current_price: float) -> bool:
        """Update trailing stop and check if triggered"""
        if asset not in self.trailing_stops:
            return False
            
        stop_data = self.trailing_stops[asset]
        
        if not stop_data['active']:
            return False
        
        # Update highest price
        if current_price > stop_data['highest_price']:
            stop_data['highest_price'] = current_price
            
            # Move stop up
            new_stop = current_price * (1 - stop_data['trail_percent'])
            if new_stop > stop_data['current_stop']:
                stop_data['current_stop'] = new_stop
                logger.debug(f"Trailing stop raised for {asset}: ${new_stop:.2f}")
        
        # Check if stop triggered
        if current_price <= stop_data['current_stop']:
            logger.info(f"Trailing stop triggered for {asset}: "
                       f"Price=${current_price:.2f}, Stop=${stop_data['current_stop']:.2f}")
            stop_data['active'] = False
            return True
        
        return False
    
    def calculate_drawdown(self, portfolio_value: float):
        """Calculate current drawdown"""
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value
        
        current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital if self.peak_capital > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record drawdown history
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'peak_capital': self.peak_capital,
            'drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown
        })
        
        return current_drawdown
    
    def check_drawdown_limit(self, portfolio_value: float, max_drawdown_limit: float = 0.10) -> bool:
        """Check if drawdown exceeds limit"""
        drawdown = self.calculate_drawdown(portfolio_value)
        
        if drawdown > max_drawdown_limit:
            logger.warning(f"Drawdown limit exceeded: {drawdown:.2%} > {max_drawdown_limit:.2%}")
            return True
        
        return False
    
    def calculate_correlation_risk(self, positions: Dict[str, float], 
                                 correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio correlation risk"""
        if not positions or len(positions) == 1:
            return 0
        
        # Convert positions to weights
        total_value = sum(positions.values())
        if total_value == 0:
            return 0
        
        weights = {asset: value/total_value for asset, value in positions.items()}
        
        # Calculate portfolio variance from correlation matrix
        portfolio_variance = 0
        assets = list(weights.keys())
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if asset1 in correlation_matrix.columns and asset2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[asset1, asset2]
                    portfolio_variance += weights[asset1] * weights[asset2] * corr
        
        # Risk score: higher correlation = higher risk
        correlation_risk = portfolio_variance / len(positions) if len(positions) > 0 else 0
        
        return correlation_risk
    
    def adjust_position_sizes(self, positions: Dict[str, float], 
                            volatilities: Dict[str, float],
                            max_position_size: float = 0.3) -> Dict[str, float]:
        """Adjust position sizes based on volatility"""
        if not positions:
            return {}
        
        total_value = sum(positions.values())
        if total_value == 0:
            return positions
        
        adjusted_positions = {}
        
        for asset, position_value in positions.items():
            volatility = volatilities.get(asset, 0.02)  # Default 2% volatility
            
            # Inverse volatility weighting: lower volatility = larger position
            inverse_vol_weight = 1 / max(volatility, 0.01)  # Avoid division by zero
            
            # Calculate new position size
            new_position_value = position_value * inverse_vol_weight
            
            # Cap at maximum position size
            max_allowed = total_value * max_position_size
            new_position_value = min(new_position_value, max_allowed)
            
            adjusted_positions[asset] = new_position_value
        
        # Normalize to maintain total value
        total_adjusted = sum(adjusted_positions.values())
        if total_adjusted > 0:
            normalization_factor = total_value / total_adjusted
            adjusted_positions = {k: v * normalization_factor for k, v in adjusted_positions.items()}
        
        return adjusted_positions
    
    def generate_risk_report(self, portfolio_value: float, positions: Dict[str, float]) -> Dict[str, any]:
        """Generate comprehensive risk report"""
        drawdown = self.calculate_drawdown(portfolio_value)
        
        report = {
            'portfolio_value': portfolio_value,
            'peak_capital': self.peak_capital,
            'current_drawdown': drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_limit_exceeded': self.check_drawdown_limit(portfolio_value),
            'num_positions': len([p for p in positions.values() if p > 0]),
            'position_concentration': max(positions.values()) / portfolio_value if portfolio_value > 0 else 0,
            'active_trailing_stops': len([s for s in self.trailing_stops.values() if s['active']]),
            'risk_score': self.calculate_overall_risk_score(portfolio_value, positions),
            'timestamp': datetime.now()
        }
        
        return report
    
    def calculate_overall_risk_score(self, portfolio_value: float, positions: Dict[str, float]) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        risk_score = 0
        
        # Drawdown component (0-40)
        drawdown = self.calculate_drawdown(portfolio_value)
        risk_score += min(drawdown * 400, 40)  # 10% drawdown = 40 points
        
        # Concentration component (0-30)
        if portfolio_value > 0:
            concentration = max(positions.values()) / portfolio_value if positions else 0
            risk_score += min(concentration * 100, 30)  # 30% concentration = 30 points
        
        # Number of positions component (0-30)
        num_positions = len([p for p in positions.values() if p > 0])
        if num_positions == 0:
            risk_score += 0
        elif num_positions == 1:
            risk_score += 30
        elif num_positions == 2:
            risk_score += 15
        else:
            risk_score += 5
        
        return min(risk_score, 100)


class PortfolioRebalancer:
    """Automatic portfolio rebalancing"""
    
    def __init__(self, target_allocations: Dict[str, float], rebalance_threshold: float = 0.05):
        self.target_allocations = target_allocations
        self.rebalance_threshold = rebalance_threshold
        self.last_rebalance = None
        
    def check_rebalance_needed(self, current_allocations: Dict[str, float]) -> bool:
        """Check if rebalancing is needed"""
        if not current_allocations or not self.target_allocations:
            return False
        
        for asset, target_weight in self.target_allocations.items():
            current_weight = current_allocations.get(asset, 0)
            
            deviation = abs(current_weight - target_weight) / target_weight if target_weight > 0 else 0
            
            if deviation > self.rebalance_threshold:
                return True
        
        return False
    
    def calculate_rebalance_trades(self, current_positions: Dict[str, float], 
                                 total_portfolio_value: float) -> Dict[str, float]:
        """Calculate trades needed to rebalance"""
        if total_portfolio_value <= 0:
            return {}
        
        rebalance_trades = {}
        
        for asset, target_weight in self.target_allocations.items():
            target_value = total_portfolio_value * target_weight
            current_value = current_positions.get(asset, 0)
            
            trade_amount = target_value - current_value
            
            if abs(trade_amount) > total_portfolio_value * 0.001:  # Minimum trade size 0.1%
                rebalance_trades[asset] = trade_amount
        
        return rebalance_trades
    
    def execute_rebalance(self, portfolio, rebalance_trades: Dict[str, float]):
        """Execute rebalance trades"""
        if not rebalance_trades:
            return
        
        logger.info("Executing portfolio rebalance...")
        
        for asset, trade_amount in rebalance_trades.items():
            if trade_amount > 0:
                # Buy
                logger.info(f"Rebalance BUY {asset}: ${trade_amount:.2f}")
            elif trade_amount < 0:
                # Sell
                logger.info(f"Rebalance SELL {asset}: ${-trade_amount:.2f}")
        
        self.last_rebalance = datetime.now()
        
    def get_rebalance_summary(self) -> Dict[str, any]:
        """Get rebalancing summary"""
        return {
            'last_rebalance': self.last_rebalance,
            'target_allocations': self.target_allocations,
            'rebalance_threshold': self.rebalance_threshold,
            'time_since_last_rebalance': datetime.now() - self.last_rebalance if self.last_rebalance else None
        }