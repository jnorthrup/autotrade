#!/usr/bin/env python3
"""
Enhanced Paper Trading Model with Web Dashboard
- Multi-asset portfolio
- ML-enhanced strategies  
- Advanced risk management
- Real-time web dashboard
"""

import sys
import os
import time
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string
import threading
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_client import CoinbaseAdvancedClient
from multi_asset_trading import MultiAssetPaperTrading
from enhanced_strategies import EnhancedTradingStrategies, PortfolioAllocator
from advanced_risk import AdvancedRiskManager, PortfolioRebalancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTradingModel:
    """Enhanced trading model with all advanced features"""
    
    def __init__(self, initial_balance=100.0, sandbox=True):
        self.client = CoinbaseAdvancedClient(sandbox=sandbox)
        self.paper_trading = MultiAssetPaperTrading(initial_balance)
        self.enhanced_strategies = EnhancedTradingStrategies()
        self.risk_manager = AdvancedRiskManager(initial_balance)
        self.portfolio_allocator = PortfolioAllocator(['BTC', 'ETH', 'ADA', 'SOL'], initial_balance)
        
        self.rebalancer = PortfolioRebalancer(
            target_allocations={'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.2, 'SOL': 0.1},
            rebalance_threshold=0.05
        )
        
        self.running = False
        self.portfolio_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
    def get_market_data(self, asset: str, days: int = 30) -> pd.DataFrame:
        """Get historical market data for asset"""
        try:
            product_id = f"{asset}-USD"
            return self.client.get_historical_data(product_id, days)
        except Exception as e:
            logger.error(f"Failed to get market data for {asset}: {e}")
            return pd.DataFrame()
    
    def analyze_asset(self, asset: str) -> Dict[str, any]:
        """Analyze asset and generate trading signals"""
        data = self.get_market_data(asset, days=60)
        
        if data.empty or len(data) < 50:
            logger.warning(f"Insufficient data for {asset}")
            return {'signal': 'HOLD', 'strength': 0, 'confidence': 0}
        
        # Generate composite signal
        signal = self.enhanced_strategies.generate_composite_signal(data)
        
        # Add market sentiment
        sentiment = self.enhanced_strategies.calculate_market_sentiment(data)
        signal['market_sentiment'] = sentiment
        
        # Calculate support/resistance
        sr_levels = self.enhanced_strategies.calculate_support_resistance(data['close'])
        signal['support_resistance'] = sr_levels
        
        return signal
    
    def calculate_position_size(self, asset: str, portfolio_value: float) -> float:
        """Calculate position size with risk management"""
        asset_price = self.paper_trading.assets[asset].price
        
        # Get portfolio metrics
        metrics = self.paper_trading.calculate_portfolio_metrics()
        current_weights = metrics.get('portfolio_weights', {})
        
        # Check correlation risk
        correlation_risk = metrics.get('correlation_risk', 0)
        
        # Adjust risk based on correlation
        base_risk = portfolio_value * 0.02  # 2% base risk
        if correlation_risk > 0.7:
            base_risk *= 0.5  # Reduce risk for high correlation
        
        # Calculate position size
        position_value = base_risk / 0.02  # 2% stop loss
        position_size = position_value / asset_price
        
        # Limit to 25% of portfolio per asset
        max_position = portfolio_value * 0.25 / asset_price
        
        return min(position_size, max_position)
    
    def execute_asset_trade(self, asset: str, signal: Dict[str, any]):
        """Execute trade for specific asset"""
        portfolio_value = self.paper_trading.get_portfolio_value()
        current_price = self.paper_trading.assets[asset].price
        
        # Check current position
        current_position = self.paper_trading.positions.get(asset, {})
        current_size = current_position.get('size', 0)
        
        logger.info(f"{asset}: Signal={signal['signal']}, Strength={signal['strength']:.2f}, "
                   f"Position={current_size:.6f}, Price=${current_price:.2f}")
        
        if signal['signal'] == 'BUY' and signal['strength'] > 10:
            if current_size == 0:
                position_size = self.calculate_position_size(asset, portfolio_value)
                if position_size > 0.0001:  # Minimum size
                    success = self.paper_trading.buy(asset, current_price, position_size)
                    if success:
                        # Set trailing stop
                        self.risk_manager.set_trailing_stop(
                            asset, current_price, stop_percent=0.02, trail_percent=0.01
                        )
        
        elif signal['signal'] == 'SELL':
            if current_size > 0:
                self.paper_trading.sell(asset, current_price, current_size)
                # Remove trailing stop
                if asset in self.risk_manager.trailing_stops:
                    self.risk_manager.trailing_stops[asset]['active'] = False
    
    def check_trailing_stops(self):
        """Check and execute trailing stops"""
        for asset, position in self.paper_trading.positions.items():
            if asset in self.risk_manager.trailing_stops:
                current_price = self.paper_trading.assets[asset].price
                triggered = self.risk_manager.update_trailing_stop(asset, current_price)
                
                if triggered and position['size'] > 0:
                    logger.info(f"Trailing stop triggered for {asset}, executing sell")
                    self.paper_trading.sell(asset, current_price, position['size'])
    
    def rebalance_portfolio_if_needed(self):
        """Check and execute portfolio rebalancing"""
        metrics = self.paper_trading.calculate_portfolio_metrics()
        portfolio_value = metrics['portfolio_value']
        
        # Calculate current allocations
        current_allocations = {}
        for asset in self.paper_trading.assets:
            position_value = self.paper_trading.positions.get(asset, {}).get('size', 0) * self.paper_trading.assets[asset].price
            current_allocations[asset] = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Check if rebalancing needed
        if self.rebalancer.check_rebalance_needed(current_allocations):
            logger.info("Portfolio rebalancing needed")
            self.paper_trading.rebalance_portfolio(self.rebalancer.target_allocations)
    
    def update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        portfolio_summary = self.paper_trading.get_performance_summary()
        
        # Calculate additional metrics
        portfolio_value = portfolio_summary['current_balance']
        drawdown = self.risk_manager.calculate_drawdown(portfolio_value)
        
        # Sharpe ratio (simplified)
        if len(self.portfolio_history) > 1:
            returns = [h['portfolio_value'] for h in self.portfolio_history[-20:]]
            if len(returns) > 1:
                daily_returns = np.diff(returns) / returns[:-1]
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        self.performance_metrics = {
            **portfolio_summary,
            'sharpe_ratio': sharpe_ratio,
            'current_drawdown': drawdown,
            'max_drawdown': self.risk_manager.max_drawdown,
            'risk_score': self.risk_manager.calculate_overall_risk_score(
                portfolio_value, 
                {asset: pos['size'] * self.paper_trading.assets[asset].price 
                 for asset, pos in self.paper_trading.positions.items()}
            ),
            'diversification_score': portfolio_summary['diversification_score'],
            'correlation_risk': portfolio_summary['correlation_risk'],
            'timestamp': datetime.now()
        }
        
        return self.performance_metrics
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING TRADING CYCLE")
        logger.info("="*60)
        
        # Update asset prices
        self.paper_trading.update_asset_prices()
        
        # Check trailing stops
        self.check_trailing_stops()
        
        # Check portfolio rebalancing
        self.rebalance_portfolio_if_needed()
        
        # Analyze and trade each asset
        for asset in self.paper_trading.assets:
            signal = self.analyze_asset(asset)
            self.execute_asset_trade(asset, signal)
        
        # Update metrics
        metrics = self.update_performance_metrics()
        
        # Log summary
        logger.info(f"Portfolio Value: ${metrics['current_balance']:.2f}")
        logger.info(f"Total Return: ${metrics['total_return']:.2f} ({metrics['return_pct']:.2f}%)")
        logger.info(f"Risk Score: {metrics['risk_score']:.1f}/100")
        logger.info(f"Diversification: {metrics['diversification_score']:.2f}")
        logger.info(f"Active Positions: {metrics['num_positions']}")
        
        # Store history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': metrics['current_balance'],
            'return_pct': metrics['return_pct']
        })
        
        return metrics
    
    def run_live_trading(self, interval_minutes: int = 5):
        """Run live paper trading"""
        logger.info("Starting enhanced paper trading...")
        self.running = True
        
        cycle_count = 0
        try:
            while self.running:
                cycle_count += 1
                logger.info(f"\nCycle #{cycle_count}")
                
                metrics = self.run_trading_cycle()
                
                # Save state every 10 cycles
                if cycle_count % 10 == 0:
                    self.paper_trading.save_portfolio_state()
                
                # Wait for next cycle
                logger.info(f"Waiting {interval_minutes} minutes for next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Stopping trading model...")
            self.running = False
        
        finally:
            # Final summary
            logger.info("\n" + "="*60)
            logger.info("FINAL TRADING SUMMARY")
            logger.info("="*60)
            
            final_metrics = self.update_performance_metrics()
            for key, value in final_metrics.items():
                if key not in ['positions', 'portfolio_weights', 'current_prices']:
                    if isinstance(value, float):
                        logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    def get_dashboard_data(self) -> Dict[str, any]:
        """Get data for web dashboard"""
        metrics = self.update_performance_metrics()
        
        # Prepare chart data
        chart_data = {
            'labels': [h['timestamp'].strftime('%H:%M') for h in self.portfolio_history[-20:]],
            'values': [h['portfolio_value'] for h in self.portfolio_history[-20:]],
            'returns': [h['return_pct'] for h in self.portfolio_history[-20:]]
        }
        
        # Asset data
        asset_data = []
        for asset in self.paper_trading.assets:
            asset_obj = self.paper_trading.assets[asset]
            position = self.paper_trading.positions.get(asset, {})
            position_value = position.get('size', 0) * asset_obj.price
            
            asset_data.append({
                'symbol': asset,
                'name': asset_obj.name,
                'price': asset_obj.price,
                'change': asset_obj.volatility * 100,  # Simplified change
                'position_size': position.get('size', 0),
                'position_value': position_value,
                'weight': position_value / metrics['current_balance'] if metrics['current_balance'] > 0 else 0
            })
        
        # Trade history (last 10 trades)
        recent_trades = self.paper_trading.trades[-10:]
        
        return {
            'metrics': metrics,
            'chart_data': chart_data,
            'assets': asset_data,
            'recent_trades': recent_trades,
            'system_status': 'RUNNING' if self.running else 'STOPPED',
            'last_update': datetime.now().isoformat()
        }


# Flask Web Dashboard
app = Flask(__name__)
trading_model = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Paper Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .metric { text-align: center; padding: 10px; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .metric .label { font-size: 12px; color: #666; }
        .positive { color: green; }
        .negative { color: red; }
        .neutral { color: #666; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn-stop { background: #dc3545; }
        .btn-stop:hover { background: #c82333; }
        .assets-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
        .asset-card { padding: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        canvas { max-height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Paper Trading Dashboard</h1>
            <div>
                <span id="status" class="btn">Status: LOADING...</span>
                <button onclick="refreshData()" class="btn">Refresh</button>
                <button onclick="toggleTrading()" id="toggleBtn" class="btn">Start Trading</button>
            </div>
        </div>
        
        <div class="card">
            <h2>Portfolio Overview</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="value" id="portfolioValue">$0.00</div>
                    <div class="label">Portfolio Value</div>
                </div>
                <div class="metric">
                    <div class="value" id="totalReturn">$0.00</div>
                    <div class="label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="value" id="returnPct">0.00%</div>
                    <div class="label">Return %</div>
                </div>
                <div class="metric">
                    <div class="value" id="riskScore">0/100</div>
                    <div class="label">Risk Score</div>
                </div>
                <div class="metric">
                    <div class="value" id="sharpeRatio">0.00</div>
                    <div class="label">Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="value" id="diversification">0.00</div>
                    <div class="label">Diversification</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Portfolio Performance</h2>
            <canvas id="portfolioChart"></canvas>
        </div>
        
        <div class="card">
            <h2>Asset Allocation</h2>
            <div class="assets-grid">
                <div id="assetCards"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Trades</h2>
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Asset</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Price</th>
                        <th>P/L</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
    </div>
    
    <script>
        let portfolioChart;
        let isTrading = false;
        
        function formatCurrency(value) {
            return '$' + parseFloat(value).toFixed(2);
        }
        
        function formatPercent(value) {
            return parseFloat(value).toFixed(2) + '%';
        }
        
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('portfolioValue').textContent = formatCurrency(data.metrics.current_balance);
            document.getElementById('totalReturn').textContent = formatCurrency(data.metrics.total_return);
            document.getElementById('returnPct').textContent = formatPercent(data.metrics.return_pct);
            document.getElementById('riskScore').textContent = Math.round(data.metrics.risk_score) + '/100';
            document.getElementById('sharpeRatio').textContent = data.metrics.sharpe_ratio.toFixed(2);
            document.getElementById('diversification').textContent = data.metrics.diversification_score.toFixed(2);
            
            // Update status
            document.getElementById('status').textContent = 'Status: ' + data.system_status;
            
            // Update chart
            updateChart(data.chart_data);
            
            // Update asset cards
            updateAssetCards(data.assets);
            
            // Update trades table
            updateTradesTable(data.recent_trades);
        }
        
        function updateChart(chartData) {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            
            if (portfolioChart) {
                portfolioChart.destroy();
            }
            
            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Portfolio Value',
                        data: chartData.values,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        function updateAssetCards(assets) {
            const container = document.getElementById('assetCards');
            container.innerHTML = '';
            
            assets.forEach(asset => {
                const card = document.createElement('div');
                card.className = 'asset-card card';
                card.innerHTML = `
                    <h3>${asset.symbol} - ${asset.name}</h3>
                    <div>Price: ${formatCurrency(asset.price)}</div>
                    <div>Change: ${asset.change.toFixed(2)}%</div>
                    <div>Position: ${asset.position_size.toFixed(6)}</div>
                    <div>Value: ${formatCurrency(asset.position_value)}</div>
                    <div>Weight: ${(asset.weight * 100).toFixed(1)}%</div>
                `;
                container.appendChild(card);
            });
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('tradesBody');
            tbody.innerHTML = '';
            
            trades.slice().reverse().forEach(trade => {
                const row = document.createElement('tr');
                const time = new Date(trade.timestamp).toLocaleTimeString();
                const plClass = trade.side === 'sell' ? (trade.profit >= 0 ? 'positive' : 'negative') : 'neutral';
                const plValue = trade.side === 'sell' ? formatCurrency(trade.profit) : '-';
                
                row.innerHTML = `
                    <td>${time}</td>
                    <td>${trade.asset}</td>
                    <td><strong class="${trade.side === 'buy' ? 'positive' : 'negative'}">${trade.side.toUpperCase()}</strong></td>
                    <td>${parseFloat(trade.size).toFixed(6)}</td>
                    <td>${formatCurrency(trade.price)}</td>
                    <td class="${plClass}">${plValue}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function refreshData() {
            axios.get('/api/dashboard')
                .then(response => {
                    updateDashboard(response.data);
                })
                .catch(error => {
                    console.error('Error fetching dashboard data:', error);
                });
        }
        
        function toggleTrading() {
            const btn = document.getElementById('toggleBtn');
            if (isTrading) {
                axios.post('/api/stop')
                    .then(() => {
                        isTrading = false;
                        btn.textContent = 'Start Trading';
                        btn.className = 'btn';
                        refreshData();
                    });
            } else {
                axios.post('/api/start')
                    .then(() => {
                        isTrading = true;
                        btn.textContent = 'Stop Trading';
                        btn.className = 'btn btn-stop';
                        refreshData();
                    });
            }
        }
        
        // Initial load
        refreshData();
        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Serve the dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    if trading_model:
        data = trading_model.get_dashboard_data()
        return jsonify(data)
    return jsonify({'error': 'Trading model not initialized'})

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading"""
    global trading_model
    if trading_model and not trading_model.running:
        # Start in background thread
        thread = threading.Thread(target=trading_model.run_live_trading, args=(5,))
        thread.daemon = True
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    global trading_model
    if trading_model:
        trading_model.running = False
    return jsonify({'status': 'stopped'})


def main():
    """Main entry point"""
    global trading_model
    
    print("\n" + "="*60)
    print("ENHANCED PAPER TRADING MODEL")
    print("Starting with $100 USD")
    print("="*60 + "\n")
    
    # Initialize trading model
    trading_model = EnhancedTradingModel(initial_balance=100.0, sandbox=True)
    
    # Run one cycle to initialize
    logger.info("Initializing trading model...")
    trading_model.run_trading_cycle()
    
    print("\nStarting web dashboard on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Start Flask app in background thread
    flask_thread = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if trading_model:
            trading_model.running = False

if __name__ == "__main__":
    main()