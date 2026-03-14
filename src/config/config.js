require('dotenv').config();

const config = {
  coinbase: {
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
    passphrase: process.env.COINBASE_PASSPHRASE,
    apiUrl: process.env.COINBASE_API_URL || 'https://api.exchange.coinbase.com',
    sandboxUrl: 'https://api-public.sandbox.pro.coinbase.com'
  },
  
  trading: {
    paperTrading: true,
    initialBalance: parseFloat(process.env.PAPER_TRADING_BALANCE) || 100,
    currency: process.env.PAPER_TRADING_CURRENCY || 'USD',
    maxPositionSize: parseFloat(process.env.MAX_POSITION_SIZE) || 0.1,
    riskPerTrade: parseFloat(process.env.RISK_PER_TRADE) || 0.02,
    tradingPair: 'BTC-USD',
    granularity: 3600 // 1 hour candles
  },
  
  strategy: {
    name: process.env.STRATEGY_NAME || 'moving_average_crossover',
    smaShortPeriod: parseInt(process.env.SMA_SHORT_PERIOD) || 20,
    smaLongPeriod: parseInt(process.env.SMA_LONG_PERIOD) || 50,
    rsiPeriod: parseInt(process.env.RSI_PERIOD) || 14,
    macdFast: parseInt(process.env.MACD_FAST) || 12,
    macdSlow: parseInt(process.env.MACD_SLOW) || 26,
    macdSignal: parseInt(process.env.MACD_SIGNAL) || 9
  },
  
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    file: process.env.LOG_FILE || 'trading.log'
  }
};

module.exports = config;