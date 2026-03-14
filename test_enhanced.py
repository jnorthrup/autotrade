#!/usr/bin/env python3
"""
Quick Test Script for Enhanced Features
"""

import sys
import os
import time
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_asset_trading import MultiAssetPaperTrading
    from advanced_risk import AdvancedRiskManager
    from simplified_enhanced import SimplifiedEnhancedStrategies
    import numpy as np
        
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def quick_test():
    """Run quick test of enhanced features"""
    print("\n" + "="*60)
    print("QUICK TEST - ENHANCED PAPER TRADING")
    print("="*60 + "\n")
    
    try:
        # Initialize components
        paper_trading = MultiAssetPaperTrading(initial_balance=100.0)
        enhanced_strategies = SimplifiedEnhancedStrategies()
        risk_manager = AdvancedRiskManager(initial_capital=100.0)
        
        print("Testing multi-asset portfolio...")
        
        # Run 5 cycles
        for i in range(5):
            print(f"\nCycle {i+1}:")
            
            # Update asset prices
            paper_trading.update_asset_prices()
            
            # Calculate portfolio metrics
            metrics = paper_trading.calculate_portfolio_metrics()
            
            print(f"  Portfolio Value: ${metrics['portfolio_value']:.2f}")
            print(f"  Return: {metrics['return_pct']:.2f}%")
            print(f"  Diversification: {metrics['diversification_score']:.2f}")
            print(f"  Positions: {metrics['num_assets']}")
            
            # Show asset prices
            print("  Asset Prices:")
            for symbol, asset in paper_trading.assets.items():
                print(f"    {symbol}: ${asset.price:.2f}")
            
            time.sleep(1)
        
        print("\n" + "="*60)
        print("TESTING ENHANCED STRATEGIES")
        print("="*60)
        
        # Test enhanced strategies
        print("\nGenerating sample market data...")
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = []
        base_price = 45000
        
        for _ in range(len(dates)):
            change = np.random.normal(0, 0.02)
            base_price = max(base_price * (1 + change), 1000)
            prices.append(base_price)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        })
        df.set_index('date', inplace=True)
        
        # Test enhanced strategies
        signal = enhanced_strategies.generate_composite_signal(df)
        sentiment = enhanced_strategies.calculate_market_sentiment(df)
        
        print(f"\nSignal: {signal.get('signal', 'HOLD')}")
        print(f"Strength: {signal.get('strength', 0):.2f}")
        print(f"Market Sentiment: {sentiment.get('overall_sentiment', 'NEUTRAL')}")
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        
        # Final summary
        final_summary = paper_trading.get_performance_summary()
        print(f"\nFinal Portfolio Value: ${final_summary['current_balance']:.2f}")
        print(f"Total Return: ${final_summary['total_return']:.2f} ({final_summary['return_pct']:.2f}%)")
        print(f"Diversification Score: {final_summary['diversification_score']:.2f}")
        print(f"Correlation Risk: {final_summary['correlation_risk']:.2f}")
        
        print("\nTo run simple paper trading:")
        print("  python3 run_paper_trading.py")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import pandas as pd
    quick_test()