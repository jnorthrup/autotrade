import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from coinbase_client import CoinbaseClient
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.client = CoinbaseClient()
        self.paper_trading = PaperTrading(Config.INITIAL_BALANCE)
        self.strategies = {
            'ma_crossover': MovingAverageCrossover(self.paper_trading),
            'rsi': RSIStrategy(self.paper_trading),
            'bollinger': BollingerBandsStrategy(self.paper_trading),
            'macd': MACDStrategy(self.paper_trading)
        }
        self.active_strategy = 'ma_crossover'
        self.running = False
        
    def get_market_data(self, days=30):
        """Fetch historical market data"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # For paper trading, we'll simulate getting historical data
            # In real implementation, this would use the Coinbase API
            
            # Generate simulated data for demonstration
            dates = pd.date_range(start=start_time, end=end_time, freq='1H')
            prices = []
            current_price = 50000  # Starting price
            
            for _ in range(len(dates)):
                # Simulate price movement with trend and volatility
                change = np.random.normal(0, 500)  # Random daily change
                current_price = max(current_price + change, 1000)  # Ensure positive price
                prices.append(current_price)
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': [p * (1 + np.random.uniform(-0.001, 0.001)) for p in prices],
                'high': [p * (1 + np.random.uniform(0, 0.002)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.002)) for p in prices],
                'close': prices,
                'volume': [np.random.uniform(100, 1000) for _ in prices]
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def execute_strategy(self, data: pd.DataFrame):
        """Execute trading strategy based on signals"""
        strategy = self.strategies[self.active_strategy]
        signal = strategy.generate_signals(data)
        
        current_price = data['close'].iloc[-1]
        self.paper_trading.current_price = current_price
        
        portfolio = self.paper_trading.get_portfolio_summary(current_price)
        btc_position = portfolio['btc_position']
        
        # Risk management
        risk_amount = portfolio['portfolio_value'] * Config.RISK_PER_TRADE
        
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Signal: {signal['signal']} (Strength: {signal['strength']:.2f})")
        logger.info(f"Portfolio Value: ${portfolio['portfolio_value']:.2f}")
        logger.info(f"BTC Position: {btc_position:.6f}")
        
        if signal['signal'] == 'BUY' and signal['strength'] > 10:
            if btc_position == 0:  # Only buy if no position
                position_size = self.paper_trading.calculate_position_size(
                    risk_amount, 
                    Config.STOP_LOSS_PERCENTAGE
                )
                
                if position_size > 0:
                    self.paper_trading.buy(current_price, position_size)
                    logger.info(f"BUY ORDER: {position_size:.6f} BTC @ ${current_price:.2f}")
        
        elif signal['signal'] == 'SELL' and signal['strength'] > 10:
            if btc_position > 0:  # Only sell if we have position
                self.paper_trading.sell(current_price, btc_position)
                logger.info(f"SELL ORDER: {btc_position:.6f} BTC @ ${current_price:.2f}")
    
    def run_backtest(self, data: pd.DataFrame):
        """Run backtesting simulation"""
        logger.info("Starting backtest...")
        
        # Reset paper trading for clean backtest
        self.paper_trading = PaperTrading(Config.INITIAL_BALANCE)
        
        # Process each data point
        for i in range(50, len(data)):  # Start after enough data for indicators
            historical_data = data.iloc[:i+1]
            self.execute_strategy(historical_data)
        
        # Calculate final performance
        final_portfolio = self.paper_trading.get_portfolio_summary(data['close'].iloc[-1])
        performance = self.paper_trading.get_performance_metrics()
        
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Final Portfolio Value: ${final_portfolio['portfolio_value']:.2f}")
        logger.info(f"Total Return: ${final_portfolio['total_return']:.2f} ({final_portfolio['return_pct']:.2f}%)")
        logger.info(f"Total Trades: {performance['total_trades']}")
        logger.info(f"Win Rate: {performance['win_rate']:.2%}")
        logger.info(f"Average Profit: ${performance['avg_profit']:.2f}")
        
        return {
            'final_value': final_portfolio['portfolio_value'],
            'total_return': final_portfolio['total_return'],
            'return_pct': final_portfolio['return_pct'],
            'total_trades': performance['total_trades'],
            'win_rate': performance['win_rate'],
            'avg_profit': performance['avg_profit']
        }
    
    def run_live_trading(self, interval_minutes=60):
        """Run live paper trading"""
        logger.info("Starting live paper trading...")
        self.running = True
        
        try:
            while self.running:
                # Get current market data
                data = self.get_market_data(days=7)
                if data is not None:
                    self.execute_strategy(data)
                
                # Wait for next iteration
                logger.info(f"Waiting {interval_minutes} minutes for next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Stopping trading bot...")
            self.running = False
        
        # Final summary
        current_data = self.get_market_data(days=1)
        if current_data is not None:
            final_summary = self.paper_trading.get_portfolio_summary(
                current_data['close'].iloc[-1]
            )
            logger.info("=== FINAL SUMMARY ===")
            logger.info(f"Final Balance: ${final_summary['balance_usd']:.2f}")
            logger.info(f"Final Portfolio Value: ${final_summary['portfolio_value']:.2f}")
            logger.info(f"Total Return: ${final_summary['total_return']:.2f} ({final_summary['return_pct']:.2f}%)")
    
    def switch_strategy(self, strategy_name):
        """Switch active trading strategy"""
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            logger.info(f"Switched to {strategy_name} strategy")
        else:
            logger.error(f"Invalid strategy: {strategy_name}")
    
    def get_strategies(self):
        """Get available strategies"""
        return list(self.strategies.keys())

if __name__ == "__main__":
    bot = TradingBot()
    
    # Run backtest
    data = bot.get_market_data(days=60)
    if data is not None:
        results = bot.run_backtest(data)
        print(f"Backtest Results: {results}")
    
    # Optionally run live trading
    # bot.run_live_trading(interval_minutes=1)  # Uncomment for live paper trading