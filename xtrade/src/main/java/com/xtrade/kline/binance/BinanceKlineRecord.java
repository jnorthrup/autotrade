package com.xtrade.kline.binance;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xtrade.kline.KlineBar;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSource;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class BinanceKlineRecord {
    static final String CSV_HEADER = "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore";

    private final long openTime;
    private final BigDecimal open;
    private final BigDecimal high;
    private final BigDecimal low;
    private final BigDecimal close;
    private final BigDecimal volume;
    private final long closeTime;
    private final BigDecimal quoteAssetVolume;
    private final long numberOfTrades;
    private final BigDecimal takerBuyBaseAssetVolume;
    private final BigDecimal takerBuyQuoteAssetVolume;
    private final String ignore;
    private final Map<String, String> metadata;

    BinanceKlineRecord(long openTime,
                       BigDecimal open,
                       BigDecimal high,
                       BigDecimal low,
                       BigDecimal close,
                       BigDecimal volume,
                       long closeTime,
                       BigDecimal quoteAssetVolume,
                       long numberOfTrades,
                       BigDecimal takerBuyBaseAssetVolume,
                       BigDecimal takerBuyQuoteAssetVolume,
                       String ignore,
                       Map<String, String> metadata) {
        this.openTime = openTime;
        this.open = Objects.requireNonNull(open, "open");
        this.high = Objects.requireNonNull(high, "high");
        this.low = Objects.requireNonNull(low, "low");
        this.close = Objects.requireNonNull(close, "close");
        this.volume = Objects.requireNonNull(volume, "volume");
        this.closeTime = closeTime;
        this.quoteAssetVolume = Objects.requireNonNull(quoteAssetVolume, "quoteAssetVolume");
        this.numberOfTrades = numberOfTrades;
        this.takerBuyBaseAssetVolume = Objects.requireNonNull(takerBuyBaseAssetVolume, "takerBuyBaseAssetVolume");
        this.takerBuyQuoteAssetVolume = Objects.requireNonNull(takerBuyQuoteAssetVolume, "takerBuyQuoteAssetVolume");
        this.ignore = ignore == null ? "0" : ignore;
        this.metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    long openTime() {
        return openTime;
    }

    long closeTime() {
        return closeTime;
    }

    String toCsvLine() {
        return openTime + ","
                + plain(open) + ","
                + plain(high) + ","
                + plain(low) + ","
                + plain(close) + ","
                + plain(volume) + ","
                + closeTime + ","
                + plain(quoteAssetVolume) + ","
                + numberOfTrades + ","
                + plain(takerBuyBaseAssetVolume) + ","
                + plain(takerBuyQuoteAssetVolume) + ","
                + ignore;
    }

    KlineBar toBar(KlineSeriesId seriesId, String producerId, KlineSource source, long sequence, boolean closed) {
        Map<String, String> meta = new LinkedHashMap<>(metadata);
        meta.putIfAbsent("exchangeSymbol", seriesId.symbol().replace("/", ""));
        return new KlineBar(
                seriesId,
                openTime,
                seriesId.interval().closeTimeExclusive(openTime),
                Math.max(closeTime, openTime),
                Math.max(closeTime, openTime),
                open,
                high,
                low,
                close,
                volume,
                quoteAssetVolume,
                numberOfTrades,
                takerBuyBaseAssetVolume,
                takerBuyQuoteAssetVolume,
                closed,
                sequence,
                producerId,
                source,
                meta);
    }

    static BinanceKlineRecord parseCsv(String line) {
        String trimmed = line == null ? "" : line.trim();
        if (trimmed.isEmpty() || trimmed.equalsIgnoreCase(CSV_HEADER)) {
            return null;
        }
        String[] parts = trimmed.split(",", -1);
        if (parts.length != 12) {
            return null;
        }
        try {
            return new BinanceKlineRecord(
                    Long.parseLong(parts[0].trim()),
                    decimal(parts[1]),
                    decimal(parts[2]),
                    decimal(parts[3]),
                    decimal(parts[4]),
                    decimal(parts[5]),
                    Long.parseLong(parts[6].trim()),
                    decimal(parts[7]),
                    Long.parseLong(parts[8].trim()),
                    decimal(parts[9]),
                    decimal(parts[10]),
                    parts[11].trim().isEmpty() ? "0" : parts[11].trim(),
                    Map.of());
        } catch (Exception e) {
            return null;
        }
    }

    static List<BinanceKlineRecord> parseRestPayload(String payload) {
        JsonArray root = JsonParser.parseString(payload).getAsJsonArray();
        List<BinanceKlineRecord> rows = new ArrayList<>();
        for (JsonElement element : root) {
            JsonArray row = element.getAsJsonArray();
            rows.add(new BinanceKlineRecord(
                    row.get(0).getAsLong(),
                    decimal(row.get(1).getAsString()),
                    decimal(row.get(2).getAsString()),
                    decimal(row.get(3).getAsString()),
                    decimal(row.get(4).getAsString()),
                    decimal(row.get(5).getAsString()),
                    row.get(6).getAsLong(),
                    decimal(row.get(7).getAsString()),
                    row.get(8).getAsLong(),
                    decimal(row.get(9).getAsString()),
                    decimal(row.get(10).getAsString()),
                    row.get(11).getAsString(),
                    Map.of()));
        }
        return rows;
    }

    static BinanceKlineRecord fromEventJson(JsonObject root) {
        JsonObject kline = root.has("k") ? root.getAsJsonObject("k") : root;
        Map<String, String> metadata = new LinkedHashMap<>();
        put(metadata, "eventType", root, "e");
        put(metadata, "exchangeSymbol", root, "s");
        put(metadata, "intervalId", kline, "i");
        put(metadata, "firstTradeId", kline, "f");
        put(metadata, "lastTradeId", kline, "L");
        return new BinanceKlineRecord(
                kline.get("t").getAsLong(),
                decimal(kline.get("o").getAsString()),
                decimal(kline.get("h").getAsString()),
                decimal(kline.get("l").getAsString()),
                decimal(kline.get("c").getAsString()),
                decimal(kline.get("v").getAsString()),
                kline.get("T").getAsLong(),
                decimal(kline.get("q").getAsString()),
                kline.get("n").getAsLong(),
                decimal(kline.get("V").getAsString()),
                decimal(kline.get("Q").getAsString()),
                "0",
                metadata);
    }

    static boolean isFinalEvent(JsonObject root) {
        JsonObject kline = root.has("k") ? root.getAsJsonObject("k") : root;
        return kline.has("x") && kline.get("x").getAsBoolean();
    }

    static String eventSymbol(JsonObject root) {
        if (root.has("s")) {
            return root.get("s").getAsString();
        }
        JsonObject kline = root.has("k") ? root.getAsJsonObject("k") : root;
        return kline.get("s").getAsString();
    }

    static String eventInterval(JsonObject root) {
        JsonObject kline = root.has("k") ? root.getAsJsonObject("k") : root;
        return kline.get("i").getAsString();
    }

    private static void put(Map<String, String> metadata, String key, JsonObject object, String field) {
        if (object.has(field) && !object.get(field).isJsonNull()) {
            metadata.put(key, object.get(field).getAsString());
        }
    }

    private static BigDecimal decimal(String value) {
        return new BigDecimal(value.trim());
    }

    private static String plain(BigDecimal value) {
        BigDecimal normalized = value.stripTrailingZeros();
        if (normalized.scale() < 0) {
            normalized = normalized.setScale(0);
        }
        return normalized.toPlainString();
    }
}
