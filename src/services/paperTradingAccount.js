const config = require('../config/config');

class PaperTradingAccount {
  constructor() {
    this.balance = config.trading.initialBalance;
    this.initialBalance = config.trading.initialBalance;
    this.currency = config.trading.currency;
    this.positions = new Map(); // productId -> {quantity, avgPrice, currentPrice}
    this.orders = new Map(); // orderId -> order details
    this.tradeHistory = [];
    this.orderIdCounter = 1;
  }

  generateOrderId() {
    return `paper_${Date.now()}_${this.orderIdCounter++}`;
  }

  async placeMarketOrder(productId, side, size, price) {
    const orderId = this.generateOrderId();
    const order = {
      id: orderId,
      productId,
      side,
      type: 'market',
      size,
      price,
      status: 'pending',
      createdAt: new Date().toISOString()
    };

    this.orders.set(orderId, order);

    // Simulate order execution
    setTimeout(() => {
      this.executeOrder(orderId, price);
    }, 100);

    return order;
  }

  async executeOrder(orderId, executionPrice) {
    const order = this.orders.get(orderId);
    if (!order) return;

    order.status = 'done';
    order.doneAt = new Date().toISOString();
    order.executedPrice = executionPrice;

    const cost = order.size * executionPrice;
    const fee = cost * 0.005; // 0.5% fee

    if (order.side === 'buy') {
      if (this.balance >= (cost + fee)) {
        this.balance -= (cost + fee);
        
        if (this.positions.has(order.productId)) {
          const position = this.positions.get(order.productId);
          position.quantity += order.size;
          position.avgPrice = ((position.avgPrice * position.quantity) + cost) / (position.quantity + order.size);
        } else {
          this.positions.set(order.productId, {
            quantity: order.size,
            avgPrice: executionPrice,
            currentPrice: executionPrice
          });
        }
      }
    } else if (order.side === 'sell') {
      if (this.positions.has(order.productId)) {
        const position = this.positions.get(order.productId);
        if (position.quantity >= order.size) {
          position.quantity -= order.size;
          this.balance += (cost - fee);
          
          if (position.quantity === 0) {
            this.positions.delete(order.productId);
          }
        }
      }
    }

    const trade = {
      id: `trade_${Date.now()}`,
      orderId,
      productId: order.productId,
      side: order.side,
      size: order.size,
      price: executionPrice,
      fee,
      timestamp: new Date().toISOString()
    };

    this.tradeHistory.push(trade);
  }

  updatePositions(currentPrices) {
    for (const [productId, position] of this.positions) {
      if (currentPrices[productId]) {
        position.currentPrice = currentPrices[productId];
      }
    }
  }

  getBalance() {
    return this.balance;
  }

  getPositions() {
    return this.positions;
  }

  getOrders() {
    return this.orders;
  }

  getTradeHistory() {
    return this.tradeHistory;
  }

  getPerformance() {
    const totalValue = this.balance + this.getPositionsValue();
    const profitLoss = totalValue - this.initialBalance;
    const profitLossPercent = (profitLoss / this.initialBalance) * 100;
    
    return {
      initialBalance: this.initialBalance,
      currentBalance: this.balance,
      positionsValue: this.getPositionsValue(),
      totalValue,
      profitLoss,
      profitLossPercent,
      totalTrades: this.tradeHistory.length
    };
  }

  getPositionsValue() {
    let value = 0;
    for (const [productId, position] of this.positions) {
      value += position.quantity * position.currentPrice;
    }
    return value;
  }
}

module.exports = PaperTradingAccount;