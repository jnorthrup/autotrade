package com.xtrade.kline;

import com.xtrade.BarAwareSlippageModel;
import com.xtrade.LimitOrder;
import com.xtrade.PaperTradingEngine;
import com.xtrade.PortfolioSnapshot;
import com.xtrade.TradeRecord;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PaperTradingEngineKlineIntegrationTest {
    private static final KlineSeriesId BTC_1M = new KlineSeriesId("binance", "BTC/USD", KlineInterval.ONE_MINUTE);

    @Test
    void limitOrdersFillAgainstBarHighLowWithConfigurableSlippage() {
        PaperTradingEngine engine = new PaperTradingEngine(10_000.0, true);
        engine.setSlippageModel(new BarAwareSlippageModel(0.0010, 0.0));

        LimitOrder buy = engine.limitBuy("BTC", 101.00, 10.0);
        KlineBar fillBar = bar(0L, 100.0, 101.2, 99.0, 100.0, 1000.0, true, 1L);

        List<TradeRecord> fills = engine.evaluateLimitOrders(fillBar);

        assertEquals(1, fills.size());
        assertEquals(0, engine.getOpenOrders().size());
        assertEquals(LimitOrder.Status.FILLED, buy.getStatus());
        assertEquals(10.0, engine.getHolding("BTC"), 1e-9);
        assertEquals(100.1, fills.get(0).getPrice().doubleValue(), 1e-9);
        assertEquals(fillBar.closePrice().doubleValue(), engine.getMarketPrice("BTC/USD"), 1e-9);
    }

    @Test
    void endToEndAgentTradesAcrossDisconnectAndReconnectWithExpectedPnl() throws Exception {
        Path stateFile = Files.createTempFile("paper-kline-state", ".json");
        Files.deleteIfExists(stateFile);

        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(512);
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "paper-btc",
                "paper btc stream",
                Collections.singleton(BTC_1M),
                null));

        AtomicInteger barsSeen = new AtomicInteger();
        AtomicInteger disconnects = new AtomicInteger();
        AtomicInteger reconnects = new AtomicInteger();
        TimedEntryExitAgent agent = new TimedEntryExitAgent(barsSeen, disconnects, reconnects);

        PaperTradingEngine firstEngine = new PaperTradingEngine(10_000.0, stateFile.toString());
        firstEngine.setSlippageModel(new BarAwareSlippageModel(0.0010, 0.0));
        PaperTradingEngineKlineAdapter firstAdapter = new PaperTradingEngineKlineAdapter(firstEngine, Collections.singletonList(agent));
        KlineSubscription firstSubscription = feed.subscribe(
                KlineSubscriptionRequest.liveOnly(BTC_1M),
                firstAdapter);

        for (int i = 0; i < 60; i++) {
            producer.publish(trendingBar(i));
        }

        assertEquals(60, barsSeen.get());
        assertTrue(firstEngine.getHolding("BTC") > 0.0);
        PortfolioSnapshot midSnapshot = firstEngine.getPortfolioSnapshot();
        assertTrue(midSnapshot.getPositions().get("BTC").getUnrealizedPnl() > 0.0);
        assertEquals(159.0, firstEngine.getMarketPrice("BTC/USD"), 1e-9);
        firstSubscription.close();

        PaperTradingEngine secondEngine = new PaperTradingEngine(10_000.0, stateFile.toString());
        secondEngine.setSlippageModel(new BarAwareSlippageModel(0.0010, 0.0));
        assertEquals(firstEngine.getCashBalance(), secondEngine.getCashBalance(), 1e-9);
        assertEquals(firstEngine.getHolding("BTC"), secondEngine.getHolding("BTC"), 1e-9);
        assertNotNull(secondEngine.getPersistencePath());

        PaperTradingEngineKlineAdapter secondAdapter = new PaperTradingEngineKlineAdapter(secondEngine, Collections.singletonList(agent));
        KlineSubscription secondSubscription = feed.subscribe(
                KlineSubscriptionRequest.liveOnly(BTC_1M),
                secondAdapter);

        for (int i = 60; i < 120; i++) {
            producer.publish(trendingBar(i));
        }
        secondSubscription.close();

        assertEquals(120, barsSeen.get());
        assertEquals(2, disconnects.get());
        assertTrue(reconnects.get() >= 2);

        List<TradeRecord> history = secondEngine.getTradeHistory();
        assertEquals(2, history.size());
        TradeRecord buy = history.get(0);
        TradeRecord sell = history.get(1);
        assertEquals(TradeRecord.Side.BUY, buy.getSide());
        assertEquals(TradeRecord.Side.SELL, sell.getSide());

        double qty = 2.0;
        double buyPrice = 100.0 * 1.0010;
        double sellPrice = 201.0 * (1.0 - 0.0010);
        double buyFee = buyPrice * qty * PaperTradingEngine.TAKER_FEE_RATE;
        double sellFee = sellPrice * qty * PaperTradingEngine.TAKER_FEE_RATE;
        double expectedEndingCash = 10_000.0 - (buyPrice * qty + buyFee) + (sellPrice * qty - sellFee);

        assertEquals(buyPrice, buy.getPrice().doubleValue(), 1e-9);
        assertEquals(sellPrice, sell.getPrice().doubleValue(), 1e-9);
        assertEquals(expectedEndingCash, secondEngine.getCashBalance(), 1e-9);
        assertEquals(0.0, secondEngine.getHolding("BTC"), 1e-12);

        PortfolioSnapshot finalSnapshot = secondEngine.getPortfolioSnapshot();
        assertEquals(expectedEndingCash, finalSnapshot.getTotalPortfolioValueUsd(), 1e-9);
        assertEquals(expectedEndingCash - 10_000.0, finalSnapshot.getTotalPortfolioValueUsd() - 10_000.0, 1e-9);
    }

    private static KlineBar trendingBar(int index) {
        double close = 100.0 + index;
        return bar(index * 60_000L, close - 1.0, close + 2.0, close - 2.0, close, 1000.0, true, index + 1L);
    }

    private static KlineBar bar(long openTimeMillis,
                                double open,
                                double high,
                                double low,
                                double close,
                                double volume,
                                boolean closed,
                                long sequence) {
        return new KlineBar(
                BTC_1M,
                openTimeMillis,
                openTimeMillis + BTC_1M.interval().toMillis(),
                openTimeMillis + BTC_1M.interval().toMillis() - 1L,
                openTimeMillis + BTC_1M.interval().toMillis() - 1L,
                BigDecimal.valueOf(open),
                BigDecimal.valueOf(high),
                BigDecimal.valueOf(low),
                BigDecimal.valueOf(close),
                BigDecimal.valueOf(volume),
                BigDecimal.valueOf(close * volume),
                50L,
                BigDecimal.valueOf(volume / 2.0),
                BigDecimal.valueOf(close * volume / 2.0),
                closed,
                sequence,
                "test-producer",
                KlineSource.WEBSOCKET_MUX,
                Collections.singletonMap("exchangeSymbol", "BTCUSDT"));
    }

    private static final class TimedEntryExitAgent implements PaperTradingAgent {
        private final AtomicInteger barsSeen;
        private final AtomicInteger disconnects;
        private final AtomicInteger reconnects;

        private TimedEntryExitAgent(AtomicInteger barsSeen, AtomicInteger disconnects, AtomicInteger reconnects) {
            this.barsSeen = barsSeen;
            this.disconnects = disconnects;
            this.reconnects = reconnects;
        }

        @Override
        public void onConnected(KlineSeriesId seriesId, PaperTradingEngine engine) {
            reconnects.incrementAndGet();
        }

        @Override
        public void onBar(KlineBar bar, PaperTradingEngine engine) {
            int seen = barsSeen.incrementAndGet();
            if (bar.openTimeMillis() == 0L && engine.getHolding("BTC") == 0.0) {
                engine.submitMarketOrder(bar.seriesId().symbol(), "BUY", 2.0, bar);
            }
            if (seen >= 102 && engine.getHolding("BTC") > 0.0) {
                engine.submitMarketOrder(bar.seriesId().symbol(), "SELL", engine.getHolding("BTC"), bar);
            }
        }

        @Override
        public void onDisconnected(KlineSeriesId seriesId, PaperTradingEngine engine) {
            disconnects.incrementAndGet();
        }
    }
}
