package com.xtrade;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the PaperTradingEngine.
 * Covers buy, sell, insufficient-funds rejection, P&L calculation,
 * trade history, persistence, and portfolio state queries.
 */
class PaperTradingEngineTest {

    private static final double STARTING_BALANCE = 10_000.00;
    private static final double BTC_PRICE = 50_000.00;
    private static final double ETH_PRICE = 3_000.00;
    private static final double XRP_PRICE = 1.00;
    private static final double SOL_PRICE = 100.00;
    private static final double ADA_PRICE = 0.50;

    private PaperTradingEngine engine;

    /** Temp directory for persistence tests. */
    private Path tempDir;

    @BeforeEach
    void setUp() throws IOException {
        tempDir = Files.createTempDirectory("pte-test-");
        engine = new PaperTradingEngine(STARTING_BALANCE, true); // in-memory only
        setAllPrices();
    }

    private void setAllPrices() {
        engine.updateMarketPrice("BTC/USD", BTC_PRICE);
        engine.updateMarketPrice("ETH/USD", ETH_PRICE);
        engine.updateMarketPrice("XRP/USD", XRP_PRICE);
        engine.updateMarketPrice("SOL/USD", SOL_PRICE);
        engine.updateMarketPrice("ADA/USD", ADA_PRICE);
    }

    // ==================================================================
    // Basic engine setup
    // ==================================================================

    @Test
    void engineInitializesWithStartingBalance() {
        assertEquals(STARTING_BALANCE, engine.getCashBalance(), 0.01);
        assertEquals(STARTING_BALANCE, engine.getStartingBalance(), 0.01);
    }

    @Test
    void engineInitializesZeroHoldingsForAllAssets() {
        for (String asset : PaperTradingEngine.TRACKED_ASSETS) {
            assertEquals(0.0, engine.getHolding(asset), 1e-12,
                    "Expected zero holdings for " + asset);
        }
    }

    @Test
    void engineRejectsNonPositiveStartingBalance() {
        assertThrows(IllegalArgumentException.class, () -> new PaperTradingEngine(0, true));
        assertThrows(IllegalArgumentException.class, () -> new PaperTradingEngine(-100, true));
    }

    // ==================================================================
    // Market BUY orders
    // ==================================================================

    @Test
    void marketBuyUpdatesBalancesCorrectly() {
        double qty = 0.1; // 0.1 BTC
        double grossCost = BTC_PRICE * qty;
        double fee = grossCost * PaperTradingEngine.TAKER_FEE_RATE;
        double totalCost = grossCost + fee;

        TradeRecord record = engine.marketBuy("BTC", qty);

        assertEquals("BTC/USD", record.getPair());
        assertEquals(TradeRecord.Side.BUY, record.getSide());
        assertEquals(BTC_PRICE, record.getPrice().doubleValue(), 0.01);
        assertEquals(qty, record.getQuantity().doubleValue(), 1e-12);
        assertEquals(fee, record.getFee().doubleValue(), 0.01);
        assertEquals(totalCost, record.getTotalCost().doubleValue(), 0.01);

        // Cash reduced by total cost
        assertEquals(STARTING_BALANCE - totalCost, engine.getCashBalance(), 0.01);
        // BTC holding increased
        assertEquals(qty, engine.getHolding("BTC"), 1e-12);
    }

    @Test
    void marketBuyMultipleTimesAccumulatesPosition() {
        engine.marketBuy("ETH", 1.0);
        engine.marketBuy("ETH", 1.0);

        assertEquals(2.0, engine.getHolding("ETH"), 1e-6);
        // Average cost should be ETH_PRICE for both buys
        assertEquals(ETH_PRICE, engine.getAverageCost("ETH"), 0.01);
    }

    @Test
    void marketBuyComputesWeightedAverageCost() {
        // Buy 1 ETH at 3000
        engine.marketBuy("ETH", 1.0);

        // Update price and buy 1 more at 4000
        engine.updateMarketPrice("ETH/USD", 4000.00);
        engine.marketBuy("ETH", 1.0);

        // Average cost should be (3000*1 + 4000*1) / 2 = 3500
        assertEquals(3500.0, engine.getAverageCost("ETH"), 0.01);
    }

    // ==================================================================
    // Market SELL orders
    // ==================================================================

    @Test
    void marketSellUpdatesBalancesCorrectly() {
        // First buy some SOL (cheap enough to stay well within balance)
        double buyQty = 10.0;
        engine.marketBuy("SOL", buyQty);

        double sellQty = 5.0;
        double grossProceeds = SOL_PRICE * sellQty;
        double fee = grossProceeds * PaperTradingEngine.TAKER_FEE_RATE;
        double netProceeds = grossProceeds - fee;

        double cashBefore = engine.getCashBalance();
        TradeRecord record = engine.marketSell("SOL", sellQty);

        assertEquals("SOL/USD", record.getPair());
        assertEquals(TradeRecord.Side.SELL, record.getSide());
        assertEquals(SOL_PRICE, record.getPrice().doubleValue(), 0.01);
        assertEquals(sellQty, record.getQuantity().doubleValue(), 1e-12);

        // Cash increased by net proceeds
        assertEquals(cashBefore + netProceeds, engine.getCashBalance(), 0.01);
        // SOL holding decreased
        assertEquals(buyQty - sellQty, engine.getHolding("SOL"), 1e-12);
    }

    @Test
    void marketSellFullPositionResetsHoldings() {
        engine.marketBuy("SOL", 5.0);
        engine.marketSell("SOL", 5.0);

        assertEquals(0.0, engine.getHolding("SOL"), 1e-12);
        assertEquals(0.0, engine.getAverageCost("SOL"), 1e-12);
    }

    // ==================================================================
    // Insufficient funds / holdings
    // ==================================================================

    @Test
    void marketBuyRejectsInsufficientUSD() {
        // Try to buy more BTC than we can afford
        double tooMuchQty = STARTING_BALANCE / BTC_PRICE + 1.0;

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> engine.marketBuy("BTC", tooMuchQty));
        assertTrue(ex.getMessage().contains("Insufficient USD"),
                "Expected 'Insufficient USD' message, got: " + ex.getMessage());
    }

    @Test
    void marketSellRejectsInsufficientHoldings() {
        // Try to sell BTC we don't have
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> engine.marketSell("BTC", 0.1));
        assertTrue(ex.getMessage().contains("Insufficient"),
                "Expected 'Insufficient' message, got: " + ex.getMessage());
    }

    @Test
    void marketSellRejectsSellingMoreThanHeld() {
        engine.marketBuy("XRP", 1000.0);

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> engine.marketSell("XRP", 2000.0));
        assertTrue(ex.getMessage().contains("Insufficient"),
                "Expected 'Insufficient' message, got: " + ex.getMessage());
    }

    // ==================================================================
    // submitMarketOrder (unified API)
    // ==================================================================

    @Test
    void submitMarketOrderBuyWorks() {
        TradeRecord record = engine.submitMarketOrder("BTC/USD", "BUY", 0.1);

        assertEquals(TradeRecord.Side.BUY, record.getSide());
        assertEquals(0.1, engine.getHolding("BTC"), 1e-12);
    }

    @Test
    void submitMarketOrderSellWorks() {
        engine.submitMarketOrder("ETH/USD", "BUY", 2.0);
        TradeRecord record = engine.submitMarketOrder("ETH/USD", "SELL", 1.0);

        assertEquals(TradeRecord.Side.SELL, record.getSide());
        assertEquals(1.0, engine.getHolding("ETH"), 1e-6);
    }

    @Test
    void submitMarketOrderRejectsInvalidPair() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.submitMarketOrder("DOGE/USD", "BUY", 1.0));
    }

    @Test
    void submitMarketOrderRejectsInvalidSide() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.submitMarketOrder("BTC/USD", "HODL", 1.0));
    }

    @Test
    void submitMarketOrderRejectsUnknownPairFormat() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.submitMarketOrder("BTC-EUR", "BUY", 1.0));
    }

    // ==================================================================
    // Trade history
    // ==================================================================

    @Test
    void tradeHistoryRecordsAllTrades() {
        engine.marketBuy("BTC", 0.1);
        engine.marketBuy("ETH", 1.0);
        engine.marketSell("ETH", 0.5);

        List<TradeRecord> history = engine.getTradeHistory();
        assertEquals(3, history.size());

        assertEquals(TradeRecord.Side.BUY, history.get(0).getSide());
        assertEquals("BTC/USD", history.get(0).getPair());

        assertEquals(TradeRecord.Side.BUY, history.get(1).getSide());
        assertEquals("ETH/USD", history.get(1).getPair());

        assertEquals(TradeRecord.Side.SELL, history.get(2).getSide());
        assertEquals("ETH/USD", history.get(2).getPair());
    }

    @Test
    void tradeHistoryRecordContainsTimestamp() {
        TradeRecord record = engine.marketBuy("SOL", 1.0);
        assertNotNull(record.getTimestamp());
        // Timestamp should be close to now
        long ageMs = java.time.Duration.between(record.getTimestamp(), java.time.Instant.now()).toMillis();
        assertTrue(ageMs < 5000, "Trade timestamp should be recent");
    }

    @Test
    void tradeHistoryRecordContainsFee() {
        TradeRecord record = engine.marketBuy("SOL", 10.0);
        double expectedFee = SOL_PRICE * 10.0 * PaperTradingEngine.TAKER_FEE_RATE;
        assertEquals(expectedFee, record.getFee().doubleValue(), 0.01);
    }

    // ==================================================================
    // P&L calculation
    // ==================================================================

    @Test
    void unrealizedPnLIsCorrectWhenPriceUnchanged() {
        engine.marketBuy("BTC", 0.1);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();

        // Price unchanged => unrealized P&L should be negative due to fee on buy
        // Actually unrealized P&L is (currentPrice - avgCost) * qty
        // avgCost = grossCost/qty = 50000, currentPrice = 50000 => unrealized = 0
        // But fees are tracked separately
        PortfolioPosition btcPos = snapshot.getPositions().get("BTC");
        assertEquals(0.0, btcPos.getUnrealizedPnl(), 0.01,
                "Unrealized P&L should be 0 when price unchanged (fees tracked separately)");
    }

    @Test
    void unrealizedPnLReflectsPriceIncrease() {
        engine.marketBuy("BTC", 0.1);

        // Price goes up by $1000
        engine.updateMarketPrice("BTC/USD", 51_000.00);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();

        PortfolioPosition btcPos = snapshot.getPositions().get("BTC");
        double expectedUnrealized = (51_000.00 - 50_000.00) * 0.1; // = 100
        assertEquals(expectedUnrealized, btcPos.getUnrealizedPnl(), 0.01);
    }

    @Test
    void realizedPnLIncludesFeeDeduction() {
        // Buy 1 SOL at 100
        engine.marketBuy("SOL", 1.0);

        // Sell 1 SOL at 200 (price doubled)
        engine.updateMarketPrice("SOL/USD", 200.00);
        engine.marketSell("SOL", 1.0);

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        PortfolioPosition solPos = snapshot.getPositions().get("SOL");

        // Realized P&L = (sellPrice - avgCost) * qty - sellFee
        // = (200 - 100) * 1 - (200 * 1 * 0.0026) = 100 - 0.52 = 99.48
        double sellFee = 200.0 * 1.0 * PaperTradingEngine.TAKER_FEE_RATE;
        double expectedRealized = (200.0 - 100.0) * 1.0 - sellFee;
        assertEquals(expectedRealized, solPos.getTotalRealizedPnl(), 0.01,
                "Realized P&L should account for sell fee");
    }

    @Test
    void totalPortfolioValueIncludesCashAndHoldings() {
        engine.marketBuy("ETH", 2.0);

        double expectedGrossCost = ETH_PRICE * 2.0;
        double expectedFee = expectedGrossCost * PaperTradingEngine.TAKER_FEE_RATE;
        double expectedCash = STARTING_BALANCE - expectedGrossCost - expectedFee;
        double expectedHoldingsValue = ETH_PRICE * 2.0;
        double expectedTotal = expectedCash + expectedHoldingsValue;

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        assertEquals(expectedTotal, snapshot.getTotalPortfolioValueUsd(), 0.01);
    }

    @Test
    void totalUnrealizedAndRealizedPnLAggregated() {
        // Buy and sell SOL at a profit
        engine.marketBuy("SOL", 5.0);
        engine.updateMarketPrice("SOL/USD", 150.00);
        engine.marketSell("SOL", 5.0);

        // Buy ADA and hold (price unchanged)
        engine.marketBuy("ADA", 1000.0);

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();

        // Should have some realized P&L from SOL trade
        assertTrue(snapshot.getTotalRealizedPnl() > 0, "Should have positive realized P&L from SOL trade");
        // ADA unrealized P&L should be 0 (price unchanged)
        PortfolioPosition adaPos = snapshot.getPositions().get("ADA");
        assertEquals(0.0, adaPos.getUnrealizedPnl(), 0.01);
    }

    // ==================================================================
    // Price management
    // ==================================================================

    @Test
    void updateMarketPriceRejectsNull() {
        assertThrows(NullPointerException.class,
                () -> engine.updateMarketPrice(null, 100.0));
    }

    @Test
    void updateMarketPriceRejectsNonPositive() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.updateMarketPrice("BTC/USD", 0));
        assertThrows(IllegalArgumentException.class,
                () -> engine.updateMarketPrice("BTC/USD", -100));
    }

    @Test
    void marketOrderRejectsWhenNoPriceAvailable() {
        PaperTradingEngine fresh = new PaperTradingEngine(10_000, true);
        // No prices set
        assertThrows(IllegalStateException.class,
                () -> fresh.marketBuy("BTC", 0.1));
        assertThrows(IllegalStateException.class,
                () -> fresh.marketSell("BTC", 0.1));
    }

    // ==================================================================
    // Persistence
    // ==================================================================

    @Test
    void statePersistsToFileOnTrade() throws IOException {
        String path = tempDir.resolve("portfolio.json").toString();
        PaperTradingEngine persistEngine = new PaperTradingEngine(STARTING_BALANCE, path);
        persistEngine.updateMarketPrice("BTC/USD", BTC_PRICE);
        persistEngine.marketBuy("BTC", 0.1);

        // File should exist
        assertTrue(Files.exists(Paths.get(path)), "Portfolio file should be created");

        // File should be valid JSON with expected content
        String json = new String(Files.readAllBytes(Paths.get(path)));
        assertTrue(json.contains("BTC"), "JSON should contain BTC data");
        assertTrue(json.contains("\"cashBalance\""), "JSON should contain cashBalance");
    }

    @Test
    void stateLoadsFromFileOnStartup() throws IOException {
        String path = tempDir.resolve("portfolio2.json").toString();

        // Create and trade with first engine
        PaperTradingEngine first = new PaperTradingEngine(STARTING_BALANCE, path);
        first.updateMarketPrice("ETH/USD", ETH_PRICE);
        first.marketBuy("ETH", 2.0);
        first.marketSell("ETH", 1.0);
        int expectedTradeCount = first.getTradeHistory().size();
        double expectedCash = first.getCashBalance();
        double expectedHolding = first.getHolding("ETH");

        // Create second engine pointing to same file — should load state
        PaperTradingEngine second = new PaperTradingEngine(STARTING_BALANCE, path);
        // Starting balance is ignored when loading state, so it still reflects original
        assertEquals(expectedCash, second.getCashBalance(), 0.01,
                "Cash balance should be restored from persisted state");
        assertEquals(expectedHolding, second.getHolding("ETH"), 1e-6,
                "ETH holding should be restored from persisted state");
        assertEquals(expectedTradeCount, second.getTradeHistory().size(),
                "Trade history should be restored from persisted state");
    }

    @Test
    void freshEngineWhenNoStateFileExists() throws IOException {
        String path = tempDir.resolve("nonexistent.json").toString();
        PaperTradingEngine engine = new PaperTradingEngine(STARTING_BALANCE, path);

        assertEquals(STARTING_BALANCE, engine.getCashBalance(), 0.01);
        assertEquals(0, engine.getTradeHistory().size());
    }

    // ==================================================================
    // Bulk price update
    // ==================================================================

    @Test
    void updateMarketPricesBulk() {
        PaperTradingEngine fresh = new PaperTradingEngine(STARTING_BALANCE, true);
        Map<String, Double> prices = Map.of(
                "BTC/USD", 55000.0,
                "ETH/USD", 3500.0,
                "XRP/USD", 1.2
        );
        fresh.updateMarketPrices(prices);

        assertEquals(55000.0, fresh.getMarketPrice("BTC/USD"), 0.01);
        assertEquals(3500.0, fresh.getMarketPrice("ETH/USD"), 0.01);
        assertEquals(1.2, fresh.getMarketPrice("XRP/USD"), 0.01);
    }

    // ==================================================================
    // Portfolio snapshot queries
    // ==================================================================

    @Test
    void portfolioSnapshotReturnsAllFiveAssets() {
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        assertEquals(5, snapshot.getPositions().size());
        assertTrue(snapshot.getPositions().containsKey("BTC"));
        assertTrue(snapshot.getPositions().containsKey("ETH"));
        assertTrue(snapshot.getPositions().containsKey("XRP"));
        assertTrue(snapshot.getPositions().containsKey("SOL"));
        assertTrue(snapshot.getPositions().containsKey("ADA"));
    }

    @Test
    void portfolioSnapshotReturnsTradeHistoryCopy() {
        engine.marketBuy("BTC", 0.1);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        assertEquals(1, snapshot.getTradeHistory().size());
    }

    // ==================================================================
    // Validation
    // ==================================================================

    @Test
    void rejectsUnknownAsset() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.marketBuy("DOGE", 100));
    }

    @Test
    void rejectsNonPositiveQuantity() {
        assertThrows(IllegalArgumentException.class,
                () -> engine.marketBuy("BTC", 0));
        assertThrows(IllegalArgumentException.class,
                () -> engine.marketBuy("BTC", -1));
    }
}
