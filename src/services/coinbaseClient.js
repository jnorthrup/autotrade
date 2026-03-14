const axios = require('axios');
const crypto = require('crypto');
const config = require('../config/config');

class CoinbaseClient {
  constructor() {
    this.apiKey = config.coinbase.apiKey;
    this.apiSecret = config.coinbase.apiSecret;
    this.passphrase = config.coinbase.passphrase;
    this.baseURL = config.coinbase.sandboxUrl;
  }

  generateSignature(timestamp, method, path, body = '') {
    const message = timestamp + method.toUpperCase() + path + body;
    const key = Buffer.from(this.apiSecret, 'base64');
    const hmac = crypto.createHmac('sha256', key);
    return hmac.update(message).digest('base64');
  }

  async makeRequest(method, path, body = null) {
    const timestamp = Date.now() / 1000;
    const bodyString = body ? JSON.stringify(body) : '';
    const signature = this.generateSignature(timestamp, method, path, bodyString);

    const headers = {
      'CB-ACCESS-KEY': this.apiKey,
      'CB-ACCESS-SIGN': signature,
      'CB-ACCESS-TIMESTAMP': timestamp,
      'CB-ACCESS-PASSPHRASE': this.passphrase,
      'Content-Type': 'application/json'
    };

    try {
      const response = await axios({
        method,
        url: `${this.baseURL}${path}`,
        headers,
        data: body
      });
      return response.data;
    } catch (error) {
      console.error('Coinbase API Error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getAccountBalance() {
    return this.makeRequest('GET', '/accounts');
  }

  async getProductTicker(productId) {
    return this.makeRequest('GET', `/products/${productId}/ticker`);
  }

  async getProductCandles(productId, granularity = 3600, start = null, end = null) {
    let path = `/products/${productId}/candles?granularity=${granularity}`;
    if (start) path += `&start=${start}`;
    if (end) path += `&end=${end}`;
    return this.makeRequest('GET', path);
  }

  async placeOrder(order) {
    return this.makeRequest('POST', '/orders', order);
  }

  async getOrder(orderId) {
    return this.makeRequest('GET', `/orders/${orderId}`);
  }

  async cancelOrder(orderId) {
    return this.makeRequest('DELETE', `/orders/${orderId}`);
  }

  async getOpenOrders() {
    return this.makeRequest('GET', '/orders?status=open');
  }
}

module.exports = CoinbaseClient;