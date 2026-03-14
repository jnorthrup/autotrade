#!/usr/bin/env python3
"""
Coinbase Advanced API Paper Trading Model
- Starts with $100 USD
- Uses sandbox mode for paper trading
- Multiple strategy options
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_client import CoinbaseAdvancedClient
from paper_trading import PaperTrading
from trading_strategies import (
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy
)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingModel:
    def __init__(self, initial_balance=100.0, sandbox=True):
        self.client = CoinbaseAdvancedClient(sandbox=sandbox)
        self.paper_trading = PaperTrading(initial_balance)
        
        self.strategies = {
            'ma_crossover': MovingAverageCrossover(self.paper_trading, short_window=20, long_window=50),
            'rsi': RSIStrategy(self.paper_trading, rsi_period=14, oversold=30, overbought=70),
            'bollinger': BollingerBandsStrategy(self.paper_trading, period=20, std_dev=2),
            'macd': MACDStrategy(self.paper_trading, fast=12, slow=26, signal=9)
        }
        self.active_strategy = 'ma_crossover'
        self.product_id = 'BTC-USD'
        
    def get_historical_data(self, days=30):
        """Fetch historical price data"""
        return self.client.get_historical_data(self.product_id, days)
    
    def get_current_price(self):
        """Get current BTC price"""
        ticker = self.client.get_product_ticker(self.product_id)
        return float(ticker.get('price', 0))
    
    def analyze_market(self, data):
        """Generate trading signals"""
        strategy = self.strategies[self.active_strategy]
        return strategy.generate_signals(data)
    
    def calculate_position_size(self, current_price):
        """Calculate position size based on risk management"""
        portfolio_value = self.paper_trading.balance
        
        if portfolio_value <= 0:
            return 0
            
        risk_amount = portfolio_value * Config.RISK_PER_TRADE
        
        position_value = risk_amount / Config.STOP_LOSS_PERCENTAGE
        position_size = position_value / current_price
        
        max_position_value = portfolio_value * Config.MAX_POSITION_SIZE
        max_size = max_position_value / current_price
        
        final_size = min(position_size, max_size)
        
        return max(final_size, Config.MIN_ORDER_SIZE)
    
    def execute_trade(self, signal, current_price):
        """Execute buy/sell based on signal"""
        portfolio = self.paper_trading.get_portfolio_summary(current_price)
        btc_position = portfolio['btc_position']
        
        if signal['signal'] == 'BUY' and signal['strength'] > 5:
            if btc_position == 0:
                position_size = self.calculate_position_size(current_price)
                if position_size > Config.MIN_ORDER_SIZE:
                    success = self.paper_trading.buy(current_price, position_size)
                    if success:
                        logger.info(f"BUY: {position_size:.6f} BTC @ ${current_price:.2f}")
                        
        elif signal['signal'] == 'SELL':
            if btc_position > 0:
                self.paper_trading.sell(current_price, btc_position)
                logger.info(f"SELL: {btc_position:.6f} BTC @ ${current_price:.2f}")
    
    def run_strategy(self, strategy_name, days=30, iterations=1):
        """Run trading strategy"""
        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return
            
        self.active_strategy = strategy_name
        logger.info(f"\n=== Running {strategy_name} strategy ===")
        
        for i in range(iterations):
            data = self.get_historical_data(days)
            if data.empty:
                logger.error("No data available")
                return
                
            current_price = data['close'].iloc[-1]
            self.paper_trading.current_price = current_price
            
            signal = self.analyze_market(data)
            self.execute_trade(signal, current_price)
            
            portfolio = self.paper_trading.get_portfolio_summary(current_price)
            logger.info(f"Price: ${current_price:.2f} | Signal: {signal['signal']} | "
                       f"Portfolio: ${portfolio['portfolio_value']:.2f} | "
                       f"BTC: {portfolio['btc_position']:.6f}")
            
            if iterations > 1:
                time.sleep(1)
        
        self.print_results(current_price)
    
    def print_results(self, current_price):
        """Print final results"""
        portfolio = self.paper_trading.get_portfolio_summary(current_price)
        metrics = self.paper_trading.get_performance_metrics()
        
        print("\n" + "="*50)
        print("TRADING RESULTS")
        print("="*50)
        print(f"Initial Balance:    ${Config.INITIAL_BALANCE:.2f}")
        print(f"Final Balance:      ${portfolio['balance_usd']:.2f}")
        print(f"Portfolio Value:    ${portfolio['portfolio_value']:.2f}")
        print(f"Total Return:       ${portfolio['total_return']:.2f} ({portfolio['return_pct']:.2f}%)")
        print(f"BTC Position:       {portfolio['btc_position']:.6f}")
        print("-"*50)
        print(f"Total Trades:       {metrics['total_trades']}")
        print(f"  Buys:             {metrics.get('total_buys', 0)}")
        print(f"  Sells:            {metrics.get('total_sells', 0)}")
        print(f"Win Rate:           {metrics['win_rate']:.1%}")
        print(f"Avg Profit/Trade:  ${metrics['avg_profit']:.2f}")
        print(f"Total P/L:          ${metrics['total_profit']:.2f}")
        print("="*50)


def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("COINBASE ADVANCED API PAPER TRADING")
    print(f"Initial Balance: ${Config.INITIAL_BALANCE} USD")
    print("="*50 + "\n")
    
    model = TradingModel(initial_balance=Config.INITIAL_BALANCE, sandbox=True)
    
    strategies = list(model.strategies.keys())
    logger.info(f"Available strategies: {strategies}")
    
    for strategy in strategies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {strategy.upper()}")
        logger.info(f"{'='*50}")
        
        model.paper_trading = PaperTrading(Config.INITIAL_BALANCE)
        model.run_strategy(strategy, days=30, iterations=10)
    
    print("\nDone! Paper trading simulation complete.")


if __name__ == "__main__":
    main()
