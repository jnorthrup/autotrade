package com.xtrade;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * In-memory paper trading engine that maintains a virtual portfolio.
 * <p>
 * Supports simulated market and limit order placement against fetched ticker prices.
 * Tracks per-asset balances, average cost, unrealized and realized P&L, fees, and
 * overall portfolio valuation. No real REST order calls are ever made to Kraken.
 * <p>
 * The 5 tracked crypto assets: BTC, ETH, XRP, SOL, ADA (all quoted in USD).
 * Starting balance: configurable, default 10,000 USD.
 * Simulated Kraken taker fee: 0.26%.
 */
public class PaperTradingEngine {

    private static final Logger LOG = LoggerFactory.getLogger(PaperTradingEngine.class);

    /** Simulated Kraken taker fee rate. */
    static final double TAKER_FEE_RATE = 0.0026;

    /** Default path for portfolio state persistence. */
    static final String DEFAULT_PERSISTENCE_PATH = "data/portfolio.json";

    /** Gson instance for JSON serialization, shared across all engines. */
    static final Gson GSON = new GsonBuilder()
            .setPrettyPrinting()
            .registerTypeAdapter(Instant.class, new TypeAdapter<Instant>() {
                @Override
                public void write(JsonWriter out, Instant value) throws IOException {
                    out.value(value != null ? value.toString() : null);
                }
                @Override
                public Instant read(JsonReader in) throws IOException {
                    String s = in.nextString();
                    return s != null ? Instant.parse(s) : null;
                }
            })
            .registerTypeAdapter(BigDecimal.class, new TypeAdapter<BigDecimal>() {
                @Override
                public void write(JsonWriter out, BigDecimal value) throws IOException {
                    out.value(value != null ? value.toPlainString() : null);
                }
                @Override
                public BigDecimal read(JsonReader in) throws IOException {
                    String s = in.nextString();
                    return s != null ? new BigDecimal(s) : null;
                }
            })
            .create();

    /** The 5 tracked base assets. */
    static final String[] TRACKED_ASSETS = {"BTC", "ETH", "XRP", "SOL", "ADA"};

    /** Map from base asset to its trading pair string, e.g. "BTC/USD". */
    static final Map<String, String> ASSET_PAIR_MAP;
    static {
        Map<String, String> m = new LinkedHashMap<>();
        for (String asset : TRACKED_ASSETS) {
            m.put(asset, asset + "/USD");
        }
        ASSET_PAIR_MAP = Collections.unmodifiableMap(m);
    }

    /** Initial USD balance. */
    private final double startingBalance;

    /** Path to the persistence file. */
    private final Path persistencePath;

    // ---- In-memory state ----

    /** Available cash in USD. */
    private double cashBalance;

    /** Holdings: base asset -> quantity held. */
    private final Map<String, Double> holdings = new HashMap<>();

    /** Average cost basis per asset: base asset -> weighted average cost per unit in USD. */
    private final Map<String, Double> averageCost = new HashMap<>();

    /** Total realized P&L per asset. */
    private final Map<String, Double> realizedPnl = new HashMap<>();

    /** Total fees paid per asset. */
    private final Map<String, Double> feesPaid = new HashMap<>();

    /** Latest market prices per trading pair string (e.g. "BTC/USD" -> price). */
    private final Map<String, Double> marketPrices = new HashMap<>();

    /** Open limit orders (not yet filled). */
    private final List<LimitOrder> openOrders = new ArrayList<>();

    /** Completed trade history. */
    private final List<TradeRecord> tradeHistory = new ArrayList<>();

    /**
     * Creates a PaperTradingEngine with the default starting balance of 10,000 USD.
     * State will be persisted to data/portfolio.json.
     */
    public PaperTradingEngine() {
        this(10_000.00, DEFAULT_PERSISTENCE_PATH);
    }

    /**
     * Creates a PaperTradingEngine with a specified starting USD balance.
     * State will be persisted to data/portfolio.json.
     *
     * @param startingBalance initial USD cash (must be positive)
     */
    public PaperTradingEngine(double startingBalance) {
        this(startingBalance, DEFAULT_PERSISTENCE_PATH);
    }

    /**
     * Creates a PaperTradingEngine with a specified starting USD balance and persistence path.
     * If the persistence file exists and is valid, prior state is loaded.
     * Otherwise a fresh engine is initialized.
     *
     * @param startingBalance  initial USD cash (must be positive)
     * @param persistencePath  path to the JSON state file
     */
    public PaperTradingEngine(double startingBalance, String persistencePath) {
        if (startingBalance <= 0) {
            throw new IllegalArgumentException("Starting balance must be positive, got: " + startingBalance);
        }
        Objects.requireNonNull(persistencePath, "Persistence path must not be null");

        this.startingBalance = startingBalance;
        this.persistencePath = Paths.get(persistencePath);

        // Initialize holdings to zero for all tracked assets
        for (String asset : TRACKED_ASSETS) {
            holdings.put(asset, 0.0);
            averageCost.put(asset, 0.0);
            realizedPnl.put(asset, 0.0);
            feesPaid.put(asset, 0.0);
        }

        // Attempt to load prior state from file
        if (!loadState()) {
            this.cashBalance = startingBalance;
            LOG.info("PaperTradingEngine initialized with starting balance: ${}", String.format("%.2f", startingBalance));
        }
    }

    /**
     * Creates a PaperTradingEngine without file persistence (for testing).
     * State is kept purely in-memory.
     *
     * @param startingBalance initial USD cash (must be positive)
     * @param noPersistence   dummy flag to select this constructor
     */
    public PaperTradingEngine(double startingBalance, boolean noPersistence) {
        if (startingBalance <= 0) {
            throw new IllegalArgumentException("Starting balance must be positive, got: " + startingBalance);
        }
        this.startingBalance = startingBalance;
        this.persistencePath = null;
        this.cashBalance = startingBalance;

        for (String asset : TRACKED_ASSETS) {
            holdings.put(asset, 0.0);
            averageCost.put(asset, 0.0);
            realizedPnl.put(asset, 0.0);
            feesPaid.put(asset, 0.0);
        }

        LOG.info("PaperTradingEngine initialized (in-memory only) with starting balance: ${}",
                String.format("%.2f", startingBalance));
    }

    // ======================================================================
    // Price management
    // ======================================================================

    /**
     * Updates the current market price for a given pair. This is used to evaluate
     * limit order fills and portfolio valuations.
     *
     * @param pair  trading pair string, e.g. "BTC/USD"
     * @param price current market price
     */
    public void updateMarketPrice(String pair, double price) {
        Objects.requireNonNull(pair, "Pair must not be null");
        if (price <= 0) {
            throw new IllegalArgumentException("Price must be positive, got: " + price);
        }
        marketPrices.put(pair, price);
        LOG.debug("Price updated: {} = {}", pair, price);
    }

    /**
     * Bulk-updates market prices from a map.
     *
     * @param prices map of pair string to price
     */
    public void updateMarketPrices(Map<String, Double> prices) {
        Objects.requireNonNull(prices, "Prices map must not be null");
        for (Map.Entry<String, Double> entry : prices.entrySet()) {
            updateMarketPrice(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Gets the latest market price for a pair.
     *
     * @param pair trading pair string
     * @return current price, or 0.0 if not set
     */
    public double getMarketPrice(String pair) {
        return marketPrices.getOrDefault(pair, 0.0);
    }

    // ======================================================================
    // Market orders
    // ======================================================================

    /**
     * Places a market BUY order. Executes immediately at the latest market price.
     * The total cost includes the simulated 0.26% taker fee.
     *
     * @param asset    base asset (e.g. "BTC")
     * @param quantity amount to buy
     * @return the TradeRecord for the executed trade
     * @throws IllegalStateException if no market price is available or insufficient USD
     */
    public TradeRecord marketBuy(String asset, double quantity) {
        validateAsset(asset);
        validateQuantity(quantity);
        String pair = ASSET_PAIR_MAP.get(asset);
        double price = getMarketPrice(pair);
        if (price <= 0) {
            throw new IllegalStateException("No market price available for " + pair);
        }
        return executeBuy(asset, pair, price, quantity);
    }

    /**
     * Places a market SELL order. Executes immediately at the latest market price.
     * The proceeds are reduced by the simulated 0.26% taker fee.
     *
     * @param asset    base asset (e.g. "BTC")
     * @param quantity amount to sell
     * @return the TradeRecord for the executed trade
     * @throws IllegalStateException if no market price is available or insufficient holdings
     */
    public TradeRecord marketSell(String asset, double quantity) {
        validateAsset(asset);
        validateQuantity(quantity);
        String pair = ASSET_PAIR_MAP.get(asset);
        double price = getMarketPrice(pair);
        if (price <= 0) {
            throw new IllegalStateException("No market price available for " + pair);
        }
        return executeSell(asset, pair, price, quantity);
    }

    // ======================================================================
    // Unified market order submission
    // ======================================================================

    /**
     * Submits a market order using the trading pair string and side.
     * Executes immediately against the current ticker price and updates balances.
     * <p>
     * This is the primary order submission API matching the acceptance criteria.
     *
     * @param pair   trading pair string, e.g. "BTC/USD"
     * @param side   "BUY" or "SELL" (case-insensitive)
     * @param amount quantity to trade
     * @return the TradeRecord for the executed trade
     * @throws IllegalArgumentException if pair is unknown, side is invalid, or amount is non-positive
     * @throws IllegalStateException    if no market price is available or insufficient funds
     */
    public TradeRecord submitMarketOrder(String pair, String side, double amount) {
        Objects.requireNonNull(pair, "Pair must not be null");
        Objects.requireNonNull(side, "Side must not be null");
        validateQuantity(amount);

        // Validate pair format and extract base asset
        if (!pair.endsWith("/USD")) {
            throw new IllegalArgumentException("Unsupported pair format: " + pair + ". Expected format: BTC/USD");
        }
        String asset = extractBaseAsset(pair);
        validateAsset(asset);

        // Validate side
        TradeRecord.Side orderSide;
        try {
            orderSide = TradeRecord.Side.valueOf(side.toUpperCase().trim());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid side: " + side + ". Must be BUY or SELL.");
        }

        // Validate price availability
        double price = getMarketPrice(pair);
        if (price <= 0) {
            throw new IllegalStateException("No market price available for " + pair);
        }

        // Execute based on side
        if (orderSide == TradeRecord.Side.BUY) {
            return executeBuy(asset, pair, price, amount);
        } else {
            return executeSell(asset, pair, price, amount);
        }
    }

    // ======================================================================
    // Limit orders
    // ======================================================================

    /**
     * Places a limit BUY order. Stored until the market price drops to or below the limit price.
     *
     * @param asset      base asset
     * @param limitPrice maximum price willing to pay
     * @param quantity   amount to buy
     * @return the LimitOrder object (in OPEN status)
     */
    public LimitOrder limitBuy(String asset, double limitPrice, double quantity) {
        validateAsset(asset);
        validateQuantity(quantity);
        if (limitPrice <= 0) {
            throw new IllegalArgumentException("Limit price must be positive, got: " + limitPrice);
        }
        String pair = ASSET_PAIR_MAP.get(asset);

        // Pre-check: buyer must have enough cash at limit price + fee
        double grossCost = limitPrice * quantity;
        double fee = grossCost * TAKER_FEE_RATE;
        double totalCost = grossCost + fee;
        if (totalCost > cashBalance) {
            throw new IllegalStateException(String.format(
                    "Insufficient USD for limit BUY: need $%.2f (incl. fee), have $%.2f",
                    totalCost, cashBalance));
        }

        // Reserve funds
        cashBalance -= totalCost;

        LimitOrder order = new LimitOrder(pair, TradeRecord.Side.BUY,
                BigDecimal.valueOf(limitPrice), BigDecimal.valueOf(quantity));
        openOrders.add(order);
        LOG.info("Limit BUY order placed: {} {} @ {} (orderID={})", quantity, asset, limitPrice, order.getOrderId());
        return order;
    }

    /**
     * Places a limit SELL order. Stored until the market price rises to or exceeds the limit price.
     *
     * @param asset      base asset
     * @param limitPrice minimum price willing to receive
     * @param quantity   amount to sell
     * @return the LimitOrder object (in OPEN status)
     */
    public LimitOrder limitSell(String asset, double limitPrice, double quantity) {
        validateAsset(asset);
        validateQuantity(quantity);
        if (limitPrice <= 0) {
            throw new IllegalArgumentException("Limit price must be positive, got: " + limitPrice);
        }
        String pair = ASSET_PAIR_MAP.get(asset);

        double held = holdings.get(asset);
        if (quantity > held + 1e-12) {
            throw new IllegalStateException(String.format(
                    "Insufficient %s holdings for limit SELL: need %.8f, have %.8f",
                    asset, quantity, held));
        }

        // Reserve the asset
        holdings.put(asset, held - quantity);

        LimitOrder order = new LimitOrder(pair, TradeRecord.Side.SELL,
                BigDecimal.valueOf(limitPrice), BigDecimal.valueOf(quantity));
        openOrders.add(order);
        LOG.info("Limit SELL order placed: {} {} @ {} (orderID={})", quantity, asset, limitPrice, order.getOrderId());
        return order;
    }

    /**
     * Cancels an open limit order. For BUY orders, reserved cash is returned.
     * For SELL orders, reserved assets are returned.
     *
     * @param orderId the order ID to cancel
     * @return true if the order was found and cancelled, false otherwise
     */
    public boolean cancelOrder(String orderId) {
        Iterator<LimitOrder> it = openOrders.iterator();
        while (it.hasNext()) {
            LimitOrder order = it.next();
            if (order.getOrderId().equals(orderId) && order.getStatus() == LimitOrder.Status.OPEN) {
                order.markCancelled();
                String asset = extractBaseAsset(order.getPair());

                if (order.getSide() == TradeRecord.Side.BUY) {
                    // Return reserved cash
                    double limitPrice = order.getLimitPrice().doubleValue();
                    double qty = order.getQuantity().doubleValue();
                    double grossCost = limitPrice * qty;
                    double fee = grossCost * TAKER_FEE_RATE;
                    cashBalance += grossCost + fee;
                    LOG.info("Limit BUY order cancelled, ${} returned. OrderID={}",
                            String.format("%.2f", grossCost + fee), orderId);
                } else {
                    // Return reserved asset
                    double currentHeld = holdings.get(asset);
                    holdings.put(asset, currentHeld + order.getQuantity().doubleValue());
                    LOG.info("Limit SELL order cancelled, {} {} returned. OrderID={}",
                            String.format("%.8f", order.getQuantity()), asset, orderId);
                }

                it.remove();
                return true;
            }
        }
        return false;
    }

    /**
     * Evaluates all open limit orders against current market prices and fills those
     * whose conditions are met. Should be called after price updates.
     *
     * @return list of TradeRecords for newly filled orders
     */
    public List<TradeRecord> evaluateLimitOrders() {
        List<TradeRecord> filled = new ArrayList<>();
        Iterator<LimitOrder> it = openOrders.iterator();

        while (it.hasNext()) {
            LimitOrder order = it.next();
            if (order.getStatus() != LimitOrder.Status.OPEN) continue;

            String pair = order.getPair();
            double currentPrice = marketPrices.getOrDefault(pair, 0.0);
            if (currentPrice <= 0) continue;

            boolean fill = false;
            double orderLimitPrice = order.getLimitPrice().doubleValue();
            double orderQty = order.getQuantity().doubleValue();
            if (order.getSide() == TradeRecord.Side.BUY) {
                // Buy limit fills when market price <= limit price
                fill = currentPrice <= orderLimitPrice;
            } else {
                // Sell limit fills when market price >= limit price
                fill = currentPrice >= orderLimitPrice;
            }

            if (fill) {
                String asset = extractBaseAsset(pair);
                TradeRecord record;
                if (order.getSide() == TradeRecord.Side.BUY) {
                    // Cash already reserved; execute the buy at current market price
                    // We need to handle the difference between reserved amount and actual cost
                    double reservedCash = orderLimitPrice * orderQty * (1 + TAKER_FEE_RATE);
                    double actualGrossCost = currentPrice * orderQty;
                    double actualFee = actualGrossCost * TAKER_FEE_RATE;
                    double actualTotalCost = actualGrossCost + actualFee;
                    // Refund the difference
                    cashBalance += (reservedCash - actualTotalCost);

                    record = createBuyTradeRecord(asset, pair, currentPrice, orderQty,
                            actualFee, actualTotalCost);
                } else {
                    // Asset already reserved; execute the sell at current market price
                    record = createSellTradeRecord(asset, pair, currentPrice, orderQty);
                }

                order.markFilled();
                tradeHistory.add(record);
                filled.add(record);
                it.remove();

                LOG.info("Limit order FILLED: {} {} {} @ {} (orderID={})",
                        order.getSide(), order.getQuantity(), asset, currentPrice, order.getOrderId());
            }
        }

        return filled;
    }

    /**
     * Returns all currently open (unfilled) limit orders.
     */
    public List<LimitOrder> getOpenOrders() {
        return Collections.unmodifiableList(new ArrayList<>(openOrders));
    }

    // ======================================================================
    // Portfolio queries
    // ======================================================================

    /**
     * Generates a complete portfolio snapshot with per-asset position details,
     * overall valuation, and P&L metrics.
     *
     * @return a PortfolioSnapshot
     */
    public PortfolioSnapshot getPortfolioSnapshot() {
        Map<String, PortfolioPosition> positions = new LinkedHashMap<>();
        double totalUnrealizedPnl = 0.0;
        double totalRealizedPnl = 0.0;
        double totalFeesPaid = 0.0;

        for (String asset : TRACKED_ASSETS) {
            String pair = ASSET_PAIR_MAP.get(asset);
            double qty = holdings.get(asset);
            double avgCost = averageCost.get(asset);
            double currentPrice = marketPrices.getOrDefault(pair, 0.0);
            double assetRealizedPnl = realizedPnl.get(asset);
            double assetFeesPaid = feesPaid.get(asset);

            double unrealizedPnl = 0.0;
            double unrealizedPnlPercent = 0.0;
            double usdValue = qty * currentPrice;

            if (qty > 0 && avgCost > 0) {
                unrealizedPnl = (currentPrice - avgCost) * qty;
                unrealizedPnlPercent = ((currentPrice - avgCost) / avgCost) * 100.0;
            }

            totalUnrealizedPnl += unrealizedPnl;
            totalRealizedPnl += assetRealizedPnl;
            totalFeesPaid += assetFeesPaid;

            positions.put(asset, new PortfolioPosition(
                    pair, asset, qty, avgCost, currentPrice,
                    unrealizedPnl, unrealizedPnlPercent,
                    assetFeesPaid, assetRealizedPnl, usdValue
            ));
        }

        double totalPortfolioValueUsd = cashBalance + positions.values().stream()
                .mapToDouble(PortfolioPosition::getUsdValue)
                .sum();

        PortfolioSnapshot snapshot = new PortfolioSnapshot(
                cashBalance, totalPortfolioValueUsd, totalUnrealizedPnl,
                totalRealizedPnl, totalFeesPaid, positions,
                new ArrayList<>(tradeHistory)
        );

        LOG.info("[PORTFOLIO] Cash: ${} | Total USD: ${} | Unrealized P&L: ${} | Realized P&L: ${} | Fees: ${}",
                String.format("%.2f", cashBalance),
                String.format("%.2f", totalPortfolioValueUsd),
                String.format("%.4f", totalUnrealizedPnl),
                String.format("%.4f", totalRealizedPnl),
                String.format("%.4f", totalFeesPaid));

        for (PortfolioPosition pos : positions.values()) {
            if (pos.getQuantity() > 0) {
                LOG.info("[POSITION] {} | qty={} | avgCost={} | curPrice={} | unrealizedPnl={} | usdValue={}",
                        pos.getBaseAsset(),
                        String.format("%.8f", pos.getQuantity()),
                        String.format("%.4f", pos.getAverageCost()),
                        String.format("%.4f", pos.getCurrentPrice()),
                        String.format("%.4f", pos.getUnrealizedPnl()),
                        String.format("%.4f", pos.getUsdValue()));
            }
        }

        return snapshot;
    }

    /**
     * Returns the current USD cash balance.
     */
    public double getCashBalance() {
        return cashBalance;
    }

    /**
     * Returns the quantity held of a given asset.
     */
    public double getHolding(String asset) {
        validateAsset(asset);
        return holdings.get(asset);
    }

    /**
     * Returns the average cost basis for a given asset.
     */
    public double getAverageCost(String asset) {
        validateAsset(asset);
        return averageCost.get(asset);
    }

    /**
     * Returns the complete trade history.
     */
    public List<TradeRecord> getTradeHistory() {
        return Collections.unmodifiableList(new ArrayList<>(tradeHistory));
    }

    /**
     * Returns the starting balance.
     */
    public double getStartingBalance() {
        return startingBalance;
    }

    // ======================================================================
    // Internal execution methods
    // ======================================================================

    private TradeRecord executeBuy(String asset, String pair, double price, double quantity) {
        double grossCost = price * quantity;
        double fee = grossCost * TAKER_FEE_RATE;
        double totalCost = grossCost + fee;

        if (totalCost > cashBalance) {
            throw new IllegalStateException(String.format(
                    "Insufficient USD for BUY: need $%.2f (incl. fee $%.4f), have $%.2f",
                    totalCost, fee, cashBalance));
        }

        // Update holdings
        double currentQty = holdings.get(asset);
        double currentAvgCost = averageCost.get(asset);
        double newQty = currentQty + quantity;
        double newAvgCost = (currentQty * currentAvgCost + grossCost) / newQty;

        holdings.put(asset, newQty);
        averageCost.put(asset, newAvgCost);
        cashBalance -= totalCost;
        feesPaid.put(asset, feesPaid.get(asset) + fee);

        TradeRecord record = createBuyTradeRecord(asset, pair, price, quantity, fee, totalCost);
        tradeHistory.add(record);

        LOG.info("Market BUY executed: {} {} @ ${} | fee=${} | total=${}",
                String.format("%.8f", quantity), asset, String.format("%.2f", price),
                String.format("%.4f", fee), String.format("%.2f", totalCost));

        saveState();
        return record;
    }

    private TradeRecord executeSell(String asset, String pair, double price, double quantity) {
        double held = holdings.get(asset);
        if (quantity > held + 1e-12) {
            throw new IllegalStateException(String.format(
                    "Insufficient %s holdings: need %.8f, have %.8f",
                    asset, quantity, held));
        }

        double grossProceeds = price * quantity;
        double fee = grossProceeds * TAKER_FEE_RATE;
        double netProceeds = grossProceeds - fee;

        // Calculate realized P&L for this sale
        double avgCost = averageCost.get(asset);
        double realizedPnlForTrade = (price - avgCost) * quantity - fee;

        double currentRealized = realizedPnl.get(asset);

        // Update holdings
        double newQty = held - quantity;
        holdings.put(asset, newQty);

        // Reset average cost if fully sold
        if (newQty < 1e-12) {
            averageCost.put(asset, 0.0);
            holdings.put(asset, 0.0);
        }

        cashBalance += netProceeds;
        realizedPnl.put(asset, currentRealized + realizedPnlForTrade);
        feesPaid.put(asset, feesPaid.get(asset) + fee);

        TradeRecord record = createSellTradeRecord(asset, pair, price, quantity);
        tradeHistory.add(record);

        LOG.info("Market SELL executed: {} {} @ ${} | fee=${} | net=${} | realizedPnl=${}",
                String.format("%.8f", quantity), asset, String.format("%.2f", price),
                String.format("%.4f", fee), String.format("%.2f", netProceeds),
                String.format("%.4f", realizedPnlForTrade));

        saveState();
        return record;
    }

    private TradeRecord createBuyTradeRecord(String asset, String pair, double price,
                                              double quantity, double fee, double totalCost) {
        return new TradeRecord(
                Instant.now(), pair, TradeRecord.Side.BUY,
                BigDecimal.valueOf(price), BigDecimal.valueOf(quantity),
                BigDecimal.valueOf(fee), BigDecimal.valueOf(totalCost)
        );
    }

    private TradeRecord createSellTradeRecord(String asset, String pair, double price, double quantity) {
        double grossProceeds = price * quantity;
        double fee = grossProceeds * TAKER_FEE_RATE;
        double netProceeds = grossProceeds - fee;
        return new TradeRecord(
                Instant.now(), pair, TradeRecord.Side.SELL,
                BigDecimal.valueOf(price), BigDecimal.valueOf(quantity),
                BigDecimal.valueOf(fee), BigDecimal.valueOf(netProceeds)
        );
    }

    // ======================================================================
    // Validation helpers
    // ======================================================================

    private void validateAsset(String asset) {
        Objects.requireNonNull(asset, "Asset must not be null");
        if (!ASSET_PAIR_MAP.containsKey(asset)) {
            throw new IllegalArgumentException("Unknown asset: " + asset +
                    ". Supported assets: " + ASSET_PAIR_MAP.keySet());
        }
    }

    private void validateQuantity(double quantity) {
        if (quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be positive, got: " + quantity);
        }
    }

    /**
     * Extracts the base asset symbol from a pair string like "BTC/USD" -> "BTC".
     */
    static String extractBaseAsset(String pair) {
        int idx = pair.indexOf('/');
        return idx > 0 ? pair.substring(0, idx) : pair;
    }

    // ======================================================================
    // Persistence methods
    // ======================================================================

    /**
     * Persists the current portfolio state to the JSON file.
     * Creates parent directories if they don't exist.
     * No-op if persistence is disabled (persistencePath is null).
     */
    void saveState() {
        if (persistencePath == null) return;

        try {
            // Create parent directories if needed
            Path parent = persistencePath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            // Build serializable state
            List<PortfolioState.LimitOrderState> orderStates = new ArrayList<>();
            for (LimitOrder order : openOrders) {
                orderStates.add(new PortfolioState.LimitOrderState(
                        order.getOrderId(),
                        order.getCreatedAt().toString(),
                        order.getPair(),
                        order.getSide().name(),
                        order.getLimitPrice().doubleValue(),
                        order.getQuantity().doubleValue(),
                        order.getStatus().name()
                ));
            }

            PortfolioState state = new PortfolioState(
                    startingBalance, cashBalance,
                    new HashMap<>(holdings), new HashMap<>(averageCost),
                    new HashMap<>(realizedPnl), new HashMap<>(feesPaid),
                    new HashMap<>(marketPrices),
                    new ArrayList<>(tradeHistory), orderStates
            );

            String json = GSON.toJson(state);
            Files.write(persistencePath, json.getBytes(StandardCharsets.UTF_8));

            LOG.debug("Portfolio state persisted to {}", persistencePath);
        } catch (IOException e) {
            LOG.error("Failed to persist portfolio state to {}: {}", persistencePath, e.getMessage());
        }
    }

    /**
     * Loads portfolio state from the JSON persistence file.
     * Returns true if state was successfully loaded, false otherwise.
     * No-op if persistence is disabled (persistencePath is null).
     *
     * @return true if prior state was loaded successfully
     */
    boolean loadState() {
        if (persistencePath == null) return false;

        if (!Files.exists(persistencePath)) {
            LOG.info("No prior state file found at {}", persistencePath);
            return false;
        }

        try (Reader reader = Files.newBufferedReader(persistencePath, StandardCharsets.UTF_8)) {
            PortfolioState state = GSON.fromJson(reader, PortfolioState.class);
            if (state == null) {
                LOG.warn("State file was empty or invalid: {}", persistencePath);
                return false;
            }

            // Restore state
            this.cashBalance = state.getCashBalance();

            for (String asset : TRACKED_ASSETS) {
                holdings.put(asset, state.getHoldings().getOrDefault(asset, 0.0));
                averageCost.put(asset, state.getAverageCost().getOrDefault(asset, 0.0));
                realizedPnl.put(asset, state.getRealizedPnl().getOrDefault(asset, 0.0));
                feesPaid.put(asset, state.getFeesPaid().getOrDefault(asset, 0.0));
            }

            marketPrices.clear();
            if (state.getMarketPrices() != null) {
                marketPrices.putAll(state.getMarketPrices());
            }

            tradeHistory.clear();
            if (state.getTradeHistory() != null) {
                tradeHistory.addAll(state.getTradeHistory());
            }

            openOrders.clear();
            if (state.getOpenOrders() != null) {
                for (PortfolioState.LimitOrderState los : state.getOpenOrders()) {
                    LimitOrder order = new LimitOrder(
                            los.getPair(),
                            TradeRecord.Side.valueOf(los.getSide()),
                            BigDecimal.valueOf(los.getLimitPrice()),
                            BigDecimal.valueOf(los.getQuantity())
                    );
                    if ("CANCELLED".equals(los.getStatus())) {
                        order.markCancelled();
                    } else if ("FILLED".equals(los.getStatus())) {
                        order.markFilled();
                    }
                    openOrders.add(order);
                }
            }

            LOG.info("Portfolio state loaded from {} ({} trades, {} open orders)",
                    persistencePath, tradeHistory.size(), openOrders.size());
            return true;

        } catch (Exception e) {
            LOG.error("Failed to load portfolio state from {}: {}", persistencePath, e.getMessage());
            return false;
        }
    }

    /**
     * Returns the persistence file path, or null if persistence is disabled.
     */
    public Path getPersistencePath() {
        return persistencePath;
    }
}
