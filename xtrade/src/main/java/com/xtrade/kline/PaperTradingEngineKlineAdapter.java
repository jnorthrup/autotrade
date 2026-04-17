package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;

import java.util.List;
import java.util.Objects;

/**
 * Consumer adapter that projects canonical klines into the existing paper trading engine.
 */
public final class PaperTradingEngineKlineAdapter implements KlineConsumer {
    private final PaperTradingEngine paperTradingEngine;

    public PaperTradingEngineKlineAdapter(PaperTradingEngine paperTradingEngine) {
        this.paperTradingEngine = Objects.requireNonNull(paperTradingEngine, "paperTradingEngine must not be null");
    }

    @Override
    public void onBackfill(KlineSeriesId seriesId, List<KlineBar> bars) {
        if (!bars.isEmpty()) {
            apply(bars.get(bars.size() - 1));
        }
    }

    @Override
    public void onLiveBar(KlineBar bar) {
        apply(bar);
    }

    private void apply(KlineBar bar) {
        paperTradingEngine.updateMarketPrice(bar.seriesId().symbol(), bar.closePrice().doubleValue());
    }
}
