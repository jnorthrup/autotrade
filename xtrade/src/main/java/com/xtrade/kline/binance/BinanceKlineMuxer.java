package com.xtrade.kline.binance;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xtrade.kline.DrawThruKlineFeed;
import com.xtrade.kline.KlineInterval;
import com.xtrade.kline.KlineProducerHandle;
import com.xtrade.kline.KlineProducerRegistration;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSource;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class BinanceKlineMuxer {
    private final DrawThruKlineFeed feed;
    private final String producerId;
    private final KlineProducerHandle producer;
    private final Map<String, KlineSeriesId> byExchangeKey = new LinkedHashMap<>();
    private long sequence = 1L;

    public BinanceKlineMuxer(DrawThruKlineFeed feed, String producerId, Collection<KlineSeriesId> series) {
        this.feed = feed;
        this.producerId = producerId;
        Set<KlineSeriesId> published = new LinkedHashSet<>(series);
        for (KlineSeriesId id : published) {
            byExchangeKey.put(exchangeKey(id), id);
        }
        this.producer = feed.registerProducer(new KlineProducerRegistration(producerId, "binance websocket muxer", published, null));
    }

    public void publishEventJson(String json) {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        String exchangeSymbol = BinanceKlineRecord.eventSymbol(root);
        String interval = BinanceKlineRecord.eventInterval(root);
        KlineSeriesId seriesId = byExchangeKey.get(exchangeSymbol + ":" + interval);
        if (seriesId == null) {
            throw new IllegalArgumentException("No registered series for " + exchangeSymbol + " interval " + interval);
        }
        BinanceKlineRecord record = BinanceKlineRecord.fromEventJson(root);
        producer.publish(record.toBar(seriesId, producerId, KlineSource.WEBSOCKET_MUX, sequence++, BinanceKlineRecord.isFinalEvent(root)));
    }

    public void publishRecorded(Path jsonLinesFile) throws IOException {
        for (String line : Files.readAllLines(jsonLinesFile, StandardCharsets.UTF_8)) {
            String trimmed = line.trim();
            if (!trimmed.isEmpty()) {
                publishEventJson(trimmed);
            }
        }
    }

    public static List<KlineSeriesId> seriesIds(Collection<String> exchangeSymbols, KlineInterval interval) {
        List<KlineSeriesId> ids = new ArrayList<>();
        for (String symbol : exchangeSymbols) {
            ids.add(new KlineSeriesId("binance", splitSymbol(symbol), interval));
        }
        return ids;
    }

    private static String exchangeKey(KlineSeriesId id) {
        return id.symbol().replace("/", "") + ":" + id.interval().wireName();
    }

    private static String splitSymbol(String exchangeSymbol) {
        if (exchangeSymbol.endsWith("USDT")) {
            return exchangeSymbol.substring(0, exchangeSymbol.length() - 4) + "/USDT";
        }
        if (exchangeSymbol.endsWith("BUSD")) {
            return exchangeSymbol.substring(0, exchangeSymbol.length() - 4) + "/BUSD";
        }
        if (exchangeSymbol.endsWith("USDC")) {
            return exchangeSymbol.substring(0, exchangeSymbol.length() - 4) + "/USDC";
        }
        throw new IllegalArgumentException("Unsupported quote asset in symbol: " + exchangeSymbol);
    }
}
