import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import Config

logger = logging.getLogger(__name__)

class PaperTrading:
    def __init__(self, initial_balance=100.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self.current_price = 0.0
        
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value including positions"""
        position_value = sum(
            position['size'] * current_price 
            for position in self.positions.values()
        )
        return self.balance + position_value
    
    def calculate_position_size(self, risk_amount, stop_loss_pct):
        """Calculate position size based on risk management"""
        if stop_loss_pct <= 0:
            return 0
        
        position_size = (risk_amount / stop_loss_pct) / self.current_price
        max_position = self.balance * Config.MAX_POSITION_SIZE / self.current_price
        
        return min(position_size, max_position)
    
    def buy(self, price, size, order_id=None):
        """Execute a buy order in paper trading mode"""
        if size <= 0:
            return False
            
        cost = price * size
        if cost > self.balance:
            logger.warning(f"Insufficient funds. Required: ${cost:.2f}, Available: ${self.balance:.2f}")
            return False
        
        # Update balance
        self.balance -= cost
        
        # Update positions
        if 'BTC' in self.positions:
            self.positions['BTC']['size'] += size
            self.positions['BTC']['cost'] += cost
        else:
            self.positions['BTC'] = {
                'size': size,
                'cost': cost,
                'entry_price': price
            }
        
        # Record order
        order = {
            'order_id': order_id or f"paper_buy_{len(self.orders)}",
            'side': 'buy',
            'size': size,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now(),
            'status': 'filled'
        }
        self.orders.append(order)
        self.trades.append({
            'side': 'buy',
            'size': size,
            'price': price,
            'cost': cost,
            'profit': 0,
            'timestamp': datetime.now()
        })
        
        logger.info(f"BUY: {size:.6f} BTC @ ${price:.2f} = ${cost:.2f}")
        return True
    
    def sell(self, price, size, order_id=None):
        """Execute a sell order in paper trading mode"""
        if 'BTC' not in self.positions or self.positions['BTC']['size'] < size:
            logger.warning("Insufficient BTC to sell")
            return False
        
        proceeds = price * size
        
        # Update balance
        self.balance += proceeds
        
        # Update positions
        self.positions['BTC']['size'] -= size
        cost_basis = (self.positions['BTC']['cost'] / 
                     (self.positions['BTC']['size'] + size)) * size
        self.positions['BTC']['cost'] -= cost_basis
        
        if self.positions['BTC']['size'] <= 0:
            del self.positions['BTC']
        
        # Calculate profit/loss
        profit = proceeds - cost_basis
        
        # Record trade
        trade = {
            'side': 'sell',
            'size': size,
            'price': price,
            'proceeds': proceeds,
            'cost_basis': cost_basis,
            'profit': profit,
            'timestamp': datetime.now()
        }
        self.trades.append(trade)
        
        logger.info(f"SELL: {size:.6f} BTC @ ${price:.2f} = ${proceeds:.2f} (Profit: ${profit:.2f})")
        return True
    
    def get_portfolio_summary(self, current_price):
        """Get current portfolio summary"""
        portfolio_value = self.get_portfolio_value(current_price)
        
        summary = {
            'balance_usd': self.balance,
            'btc_position': self.positions.get('BTC', {}).get('size', 0),
            'position_value': self.positions.get('BTC', {}).get('size', 0) * current_price,
            'portfolio_value': portfolio_value,
            'total_return': portfolio_value - self.initial_balance,
            'return_pct': ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        }
        
        return summary
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_buys': 0,
                'total_sells': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        sells = [t for t in self.trades if t['side'] == 'sell']
        profits = [trade['profit'] for trade in sells]
        winning_trades = [p for p in profits if p > 0]
        
        return {
            'total_trades': len(self.trades),
            'total_buys': len([t for t in self.trades if t['side'] == 'buy']),
            'total_sells': len(sells),
            'win_rate': len(winning_trades) / len(profits) if profits else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'total_profit': sum(profits),
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0
        }