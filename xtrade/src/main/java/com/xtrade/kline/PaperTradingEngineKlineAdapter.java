package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Consumer adapter that projects canonical klines into the paper trading engine
 * and optionally lets trading agents react to each bar.
 */
public final class PaperTradingEngineKlineAdapter implements KlineConsumer {
    private final PaperTradingEngine paperTradingEngine;
    private final List<PaperTradingAgent> agents;
    private volatile boolean connected;

    public PaperTradingEngineKlineAdapter(PaperTradingEngine paperTradingEngine) {
        this(paperTradingEngine, Collections.emptyList());
    }

    public PaperTradingEngineKlineAdapter(PaperTradingEngine paperTradingEngine, List<PaperTradingAgent> agents) {
        this.paperTradingEngine = Objects.requireNonNull(paperTradingEngine, "paperTradingEngine must not be null");
        Objects.requireNonNull(agents, "agents must not be null");
        this.agents = Collections.unmodifiableList(new ArrayList<>(agents));
    }

    @Override
    public void onBackfill(KlineSeriesId seriesId, List<KlineBar> bars) {
        if (!connected) {
            connected = true;
            notifyConnected(seriesId);
        }
        for (KlineBar bar : bars) {
            apply(bar);
        }
    }

    @Override
    public void onLiveBar(KlineBar bar) {
        if (!connected) {
            connected = true;
            notifyConnected(bar.seriesId());
        }
        apply(bar);
    }

    @Override
    public void onClosed(KlineSeriesId seriesId) {
        connected = false;
        for (PaperTradingAgent agent : agents) {
            agent.onDisconnected(seriesId, paperTradingEngine);
        }
    }

    private void notifyConnected(KlineSeriesId seriesId) {
        for (PaperTradingAgent agent : agents) {
            agent.onConnected(seriesId, paperTradingEngine);
        }
    }

    private void apply(KlineBar bar) {
        paperTradingEngine.updateFromBar(bar);
        for (PaperTradingAgent agent : agents) {
            agent.onBar(bar, paperTradingEngine);
        }
    }
}
