package com.xtrade.kline.binance;

import com.xtrade.PaperTradingEngine;
import com.xtrade.PortfolioSnapshot;
import com.xtrade.TradeRecord;
import com.xtrade.codec.Codec02MomentumTrend;
import com.xtrade.codec.Codec13RsiReversal;
import com.xtrade.kline.CodecPaperTradingAgent;
import com.xtrade.kline.DrawThruCachingKlineFeed;
import com.xtrade.kline.KlineBatchRequest;
import com.xtrade.kline.KlineInterval;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSubscriptionRequest;
import com.xtrade.kline.PaperTradingEngineKlineAdapter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CodecPaperTradingPipelineIntegrationTest {
    private static final KlineSeriesId BTC_USDT_1M = new KlineSeriesId("binance", "BTC/USDT", KlineInterval.ONE_MINUTE);

    @Test
    void binanceMuxToCacheToCodecSignalToPaperTradePipelineRunsUnderLatencyBudget() {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(512);
        BinanceKlineMuxer muxer = new BinanceKlineMuxer(feed, "binance-live-test", List.of(BTC_USDT_1M));

        PaperTradingEngine momentumEngine = new PaperTradingEngine(10_000.0, true);
        PaperTradingEngine rsiEngine = new PaperTradingEngine(10_000.0, true);
        CodecPaperTradingAgent momentumAgent = new CodecPaperTradingAgent(new Codec02MomentumTrend(), 0.40, 0.65, 512);
        CodecPaperTradingAgent rsiAgent = new CodecPaperTradingAgent(new Codec13RsiReversal(), 0.40, 0.65, 512);

        feed.subscribe(KlineSubscriptionRequest.liveOnly(BTC_USDT_1M),
                new PaperTradingEngineKlineAdapter(momentumEngine, List.of(momentumAgent)));
        feed.subscribe(KlineSubscriptionRequest.liveOnly(BTC_USDT_1M),
                new PaperTradingEngineKlineAdapter(rsiEngine, List.of(rsiAgent)));

        List<Long> tradeLatenciesNanos = new ArrayList<>();
        for (int i = 0; i < 75; i++) {
            int beforeTrades = momentumEngine.getTradeHistory().size() + rsiEngine.getTradeHistory().size();
            long started = System.nanoTime();
            muxer.publishEventJson(klineEventJson(i, priceForIndex(i)));
            long elapsed = System.nanoTime() - started;
            int afterTrades = momentumEngine.getTradeHistory().size() + rsiEngine.getTradeHistory().size();
            if (afterTrades > beforeTrades) {
                tradeLatenciesNanos.add(elapsed);
            }
        }

        assertEquals(75, feed.requestBars(KlineBatchRequest.latest(BTC_USDT_1M, 75)).size());
        assertFalse(tradeLatenciesNanos.isEmpty(), "expected at least one automated trade during the live pipeline");

        assertAgentProducedSignalsAndTrades(momentumAgent, momentumEngine);
        assertAgentProducedSignalsAndTrades(rsiAgent, rsiEngine);

        double maxTradeLatencyMillis = tradeLatenciesNanos.stream()
                .mapToLong(Long::longValue)
                .max()
                .orElseThrow() / 1_000_000.0;
        assertTrue(maxTradeLatencyMillis < 500.0,
                "exchange-bar to paper-trade latency exceeded budget: " + maxTradeLatencyMillis + "ms");
        assertTrue(momentumAgent.getMaxDecisionLatencyNanos() / 1_000_000.0 < 500.0);
        assertTrue(rsiAgent.getMaxDecisionLatencyNanos() / 1_000_000.0 < 500.0);
    }

    private static void assertAgentProducedSignalsAndTrades(CodecPaperTradingAgent agent, PaperTradingEngine engine) {
        assertEquals(75, agent.getObservations().size(), "agent should compute a signal for every live kline");
        assertTrue(agent.getObservations().stream().anyMatch(o -> o.getDirection() > 0.0),
                agent.getCodec().getStrategyName() + " should emit bullish signals");
        assertTrue(agent.getObservations().stream().anyMatch(o -> o.getDirection() < 0.0),
                agent.getCodec().getStrategyName() + " should emit bearish signals");
        assertTrue(agent.decisionCount("BUY") >= 1, agent.getCodec().getStrategyName() + " should submit buys");
        assertTrue(agent.decisionCount("SELL") >= 1, agent.getCodec().getStrategyName() + " should submit sells");
        assertTrue(engine.getTradeHistory().size() >= 2,
                agent.getCodec().getStrategyName() + " should reach paper execution automatically");
        assertTrue(engine.getTradeHistory().stream().anyMatch(t -> t.getSide() == TradeRecord.Side.BUY));
        assertTrue(engine.getTradeHistory().stream().anyMatch(t -> t.getSide() == TradeRecord.Side.SELL));

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        assertNotStartingBalance(snapshot, engine.getStartingBalance());
        assertEquals("BTC/USDT", engine.getTradeHistory().get(0).getPair());
    }

    private static void assertNotStartingBalance(PortfolioSnapshot snapshot, double startingBalance) {
        double delta = Math.abs(snapshot.getTotalPortfolioValueUsd() - startingBalance);
        assertTrue(delta > 0.0001 || Math.abs(snapshot.getTotalRealizedPnl()) > 0.0001 || Math.abs(snapshot.getTotalUnrealizedPnl()) > 0.0001,
                "portfolio should reflect updated PnL after paper execution");
    }

    private static double priceForIndex(int index) {
        if (index < 25) {
            return 120.0 - index;
        }
        if (index < 50) {
            return 96.0 + (index - 24) * 2.0;
        }
        return 146.0 - (index - 49) * 2.0;
    }

    private static String klineEventJson(int index, double closePrice) {
        long openTime = index * 60_000L;
        long closeTime = openTime + 59_999L;
        double openPrice = closePrice - 1.0;
        double highPrice = closePrice + 2.0;
        double lowPrice = closePrice - 2.0;
        double volume = 10.0 + index;
        double quoteVolume = closePrice * volume;
        return "{"
                + "\"e\":\"kline\"," 
                + "\"E\":" + closeTime + ","
                + "\"s\":\"BTCUSDT\","
                + "\"k\":{"
                + "\"t\":" + openTime + ","
                + "\"T\":" + closeTime + ","
                + "\"s\":\"BTCUSDT\","
                + "\"i\":\"1m\","
                + "\"f\":" + (index * 100 + 1) + ","
                + "\"L\":" + (index * 100 + 25) + ","
                + "\"o\":\"" + openPrice + "\","
                + "\"c\":\"" + closePrice + "\","
                + "\"h\":\"" + highPrice + "\","
                + "\"l\":\"" + lowPrice + "\","
                + "\"v\":\"" + volume + "\","
                + "\"n\":25,"
                + "\"x\":true,"
                + "\"q\":\"" + quoteVolume + "\","
                + "\"V\":\"" + (volume * 0.55) + "\","
                + "\"Q\":\"" + (quoteVolume * 0.55) + "\""
                + "}}";
    }
}
