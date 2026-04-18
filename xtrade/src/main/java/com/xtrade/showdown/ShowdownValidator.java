package com.xtrade.showdown;

import com.xtrade.codec.BaseCodecExpert;
import com.xtrade.codec.CodecFactory;
import com.xtrade.codec.IndicatorComputer;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Standalone validation runner that mirrors the Python ShowdownRunner
 * for cross-validation purposes.
 *
 * Reads a CSV tick file, runs each codec agent through it, and outputs
 * per-agent summary metrics as JSON to stdout.
 *
 * Usage: java com.xtrade.showdown.ShowdownValidator <csv_path> [num_ticks]
 */
public class ShowdownValidator {

    private static final double INITIAL_CASH = 100_000.0;
    private static final double CONVICTION_THRESHOLD = 0.4;
    private static final double POSITION_FRACTION = 0.25;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: ShowdownValidator <csv_path> [num_ticks] [--indicators]");
            System.exit(1);
        }

        String csvPath = args[0];
        int maxTicks = 200;
        boolean dumpIndicators = false;

        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("--indicators")) {
                dumpIndicators = true;
            } else {
                try {
                    maxTicks = Integer.parseInt(args[i]);
                } catch (NumberFormatException e) {
                    // ignore
                }
            }
        }

        // Load ticks from CSV
        List<Map<String, TickData.PairTick>> tickGroups = loadCsv(csvPath);
        if (tickGroups.isEmpty()) {
            System.err.println("No ticks loaded from " + csvPath);
            System.exit(1);
        }
        int tickLimit = Math.min(maxTicks, tickGroups.size());

        // Determine which codecs to run (1-24)
        List<Integer> codecIds = new ArrayList<>();
        for (int i = 1; i <= 24; i++) codecIds.add(i);

        Map<String, Object> results = new LinkedHashMap<>();
        results.put("tick_count", tickLimit);
        results.put("initial_cash", INITIAL_CASH);

        Map<String, Map<String, Object>> agentResults = new LinkedHashMap<>();
        List<Map<String, Object>> indicatorSnapshots = new ArrayList<>();

        // Run each codec independently
        for (int codecId : codecIds) {
            BaseCodecExpert codec = CodecFactory.createExpert(codecId);
            IndicatorComputer indComputer = new IndicatorComputer(200);

            double cash = INITIAL_CASH;
            Map<String, Double> holdings = new LinkedHashMap<>();
            List<double[]> costLots = new ArrayList<>(); // for each BUY
            Map<String, List<double[]>> costLotsMap = new LinkedHashMap<>();

            double realizedPnl = 0.0;
            int totalTrades = 0;

            // For indicator dump (first codec only)
            boolean dumpThisCodec = dumpIndicators && codecId == 1;

            for (int t = 0; t < tickLimit; t++) {
                Map<String, TickData.PairTick> tick = tickGroups.get(t);

                for (Map.Entry<String, TickData.PairTick> entry : tick.entrySet()) {
                    String pair = entry.getKey();
                    double price = entry.getValue().getPrice();
                    double volume = entry.getValue().getVolume();

                    // 1. Compute indicators
                    Map<String, Object> marketData = indComputer.compute(pair, price, volume);

                    // Dump indicator values for first codec on first 5 ticks
                    if (dumpThisCodec && t < 5) {
                        Map<String, Object> snap = new LinkedHashMap<>();
                        snap.put("tick", t);
                        snap.put("codec_id", codecId);
                        for (Map.Entry<String, Object> me : marketData.entrySet()) {
                            snap.put(me.getKey(), me.getValue());
                        }
                        indicatorSnapshots.add(snap);
                    }

                    // 2. Build indicator vector
                    double[] indicatorVec = buildIndicatorVec(marketData);

                    // 3. Forward
                    com.xtrade.codec.SignalResult sig = codec.forward(marketData, indicatorVec);
                    double conviction = sig.getConviction();
                    double direction = sig.getDirection();

                    // 4. Execute signal (mirror Python Agent.on_tick)
                    String action = "HOLD";
                    double size = 0.0;

                    if (conviction > CONVICTION_THRESHOLD) {
                        if (direction > 0) {
                            action = "BUY";
                            double spend = cash * POSITION_FRACTION * conviction;
                            if (spend > 0 && price > 0) {
                                size = spend / price;
                                cash -= size * price;
                                holdings.merge(pair, size, Double::sum);
                                costLotsMap.computeIfAbsent(pair, k -> new ArrayList<>())
                                        .add(new double[]{size, price});
                                totalTrades++;
                            }
                        } else if (direction < 0) {
                            action = "SELL";
                            double held = holdings.getOrDefault(pair, 0.0);
                            if (held > 0 && price > 0) {
                                double sellFrac = conviction * POSITION_FRACTION;
                                size = held * sellFrac;
                                if (size > 0) {
                                    cash += size * price;
                                    double remaining = held - size;
                                    if (remaining < 1e-12) remaining = 0.0;
                                    holdings.put(pair, remaining);

                                    // FIFO PnL
                                    List<double[]> lots = costLotsMap.getOrDefault(pair, new ArrayList<>());
                                    double rem = size;
                                    double tradePnl = 0.0;
                                    while (rem > 1e-15 && !lots.isEmpty()) {
                                        double[] lot = lots.get(0);
                                        double filled = Math.min(rem, lot[0]);
                                        tradePnl += (price - lot[1]) * filled;
                                        rem -= filled;
                                        lot[0] -= filled;
                                        if (lot[0] < 1e-15) lots.remove(0);
                                    }
                                    realizedPnl += tradePnl;
                                    totalTrades++;
                                }
                            }
                        }
                    }
                }
            }

            // Compute final portfolio value
            double holdingsValue = 0.0;
            double lastPrice = 0.0;
            if (!tickGroups.isEmpty() && tickLimit > 0) {
                Map<String, TickData.PairTick> lastTick = tickGroups.get(tickLimit - 1);
                for (Map.Entry<String, Double> he : holdings.entrySet()) {
                    TickData.PairTick pt = lastTick.get(he.getKey());
                    double p = pt != null ? pt.getPrice() : 0.0;
                    holdingsValue += he.getValue() * p;
                    lastPrice = p;
                }
            }
            double totalValue = cash + holdingsValue;
            double totalPnl = totalValue - INITIAL_CASH;
            double returnPct = (totalPnl / INITIAL_CASH) * 100.0;

            // Reset codec for next use
            codec.resetRuntimeState();
            codec.resetTradeLedger();

            Map<String, Object> agentData = new LinkedHashMap<>();
            agentData.put("agent_name", codec.getName());
            agentData.put("codec_id", codecId);
            agentData.put("initial_cash", INITIAL_CASH);
            agentData.put("final_value", totalValue);
            agentData.put("cash", cash);
            agentData.put("holdings_value", holdingsValue);
            agentData.put("total_pnl", totalPnl);
            agentData.put("return_pct", returnPct);
            agentData.put("realized_pnl", realizedPnl);
            agentData.put("trade_count", totalTrades);
            agentData.put("ticks_processed", tickLimit);

            agentResults.put(codec.getName(), agentData);
        }

        results.put("agents", agentResults);
        if (!indicatorSnapshots.isEmpty()) {
            results.put("indicator_snapshots", indicatorSnapshots);
        }

        // Output JSON to stdout
        System.out.println(toJson(results));
    }

    static double[] buildIndicatorVec(Map<String, Object> md) {
        double[] vec = new double[64];

        double price = getDouble(md, "price", 1.0);
        double safePrice = price != 0.0 ? price : 1.0;

        vec[0] = price / getDoubleNonZero(md, "sma_20", price);
        vec[1] = price / getDoubleNonZero(md, "sma_15", price);
        vec[2] = price / getDoubleNonZero(md, "ema_12", price);
        vec[3] = price / getDoubleNonZero(md, "ema_26", price);

        vec[4] = getDouble(md, "macd", 0.0) / safePrice;
        vec[5] = getDouble(md, "macd_signal", 0.0) / safePrice;
        vec[6] = getDouble(md, "macd_hist", 0.0) / safePrice;

        vec[7] = getDouble(md, "rsi", 50.0) / 100.0;

        double bbUpper = getDouble(md, "bb_upper", price);
        double bbLower = getDouble(md, "bb_lower", price);
        double bbMid = getDouble(md, "bb_mid", price);
        double bbWidth = bbUpper - bbLower;
        if (bbWidth > 0) {
            vec[8] = (price - bbLower) / bbWidth;
        } else {
            vec[8] = 0.5;
        }
        vec[9] = bbMid != 0.0 ? bbWidth / bbMid : 0.0;

        vec[10] = getDouble(md, "atr_14", 0.0) / safePrice;
        vec[11] = getDouble(md, "stoch_k", 50.0) / 100.0;
        vec[12] = getDouble(md, "stoch_d", 50.0) / 100.0;
        vec[13] = getDouble(md, "adx", 0.0) / 100.0;
        vec[14] = getDouble(md, "plus_di", 0.0) / 100.0;
        vec[15] = getDouble(md, "minus_di", 0.0) / 100.0;

        double vwap = getDouble(md, "vwap", price);
        vec[16] = vwap != 0.0 ? price / vwap : 1.0;

        vec[17] = getDouble(md, "momentum", 0.0) / 100.0;

        double avgVol = getDouble(md, "avg_volume", 0.0);
        double vol = getDouble(md, "volume", 0.0);
        vec[18] = avgVol > 0.0 ? vol / avgVol : 1.0;

        vec[19] = getDouble(md, "log_return", 0.0);

        // Slots 20-63 remain zero
        return vec;
    }

    private static double getDouble(Map<String, Object> md, String key, double def) {
        Object val = md.get(key);
        if (val == null) return def;
        if (val instanceof Number) return ((Number) val).doubleValue();
        return def;
    }

    private static double getDoubleNonZero(Map<String, Object> md, String key, double fallback) {
        double val = getDouble(md, key, fallback);
        return val != 0.0 ? val : fallback;
    }

    // Load CSV and group by timestamp
    static List<Map<String, TickData.PairTick>> loadCsv(String filepath) throws IOException {
        List<Row> rows = new ArrayList<>();
        Path path = Paths.get(filepath);

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String header = reader.readLine();
            if (header == null) return Collections.emptyList();

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",", -1);
                if (parts.length < 4) continue;

                double ts = Double.parseDouble(parts[0].trim());
                String pair = parts[1].trim();
                double price = Double.parseDouble(parts[2].trim());
                double volume = Double.parseDouble(parts[3].trim());
                rows.add(new Row(ts, pair, price, volume));
            }
        }

        List<Map<String, TickData.PairTick>> groups = new ArrayList<>();
        if (rows.isEmpty()) return groups;

        double currentTs = rows.get(0).timestamp;
        Map<String, TickData.PairTick> group = new LinkedHashMap<>();
        for (Row row : rows) {
            if (Double.compare(row.timestamp, currentTs) != 0) {
                groups.add(group);
                group = new LinkedHashMap<>();
                currentTs = row.timestamp;
            }
            group.put(row.pair, new TickData.PairTick(row.price, row.volume));
        }
        if (!group.isEmpty()) {
            groups.add(group);
        }
        return groups;
    }

    private static class Row {
        final double timestamp;
        final String pair;
        final double price;
        final double volume;

        Row(double timestamp, String pair, double price, double volume) {
            this.timestamp = timestamp;
            this.pair = pair;
            this.price = price;
            this.volume = volume;
        }
    }

    // Simple JSON serializer (no external deps)
    @SuppressWarnings("unchecked")
    static String toJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof Number) {
            double d = ((Number) obj).doubleValue();
            if (d == Math.floor(d) && !Double.isInfinite(d)) {
                return String.valueOf((long) d);
            }
            return String.valueOf(d);
        }
        if (obj instanceof Boolean) return obj.toString();
        if (obj instanceof String) return "\"" + escapeJson((String) obj) + "\"";
        if (obj instanceof Map) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<String, Object> e : ((Map<String, Object>) obj).entrySet()) {
                if (!first) sb.append(",");
                sb.append("\"").append(escapeJson(e.getKey())).append("\":");
                sb.append(toJson(e.getValue()));
                first = false;
            }
            sb.append("}");
            return sb.toString();
        }
        if (obj instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object item : (List<?>) obj) {
                if (!first) sb.append(",");
                sb.append(toJson(item));
                first = false;
            }
            sb.append("]");
            return sb.toString();
        }
        if (obj instanceof double[]) {
            double[] arr = (double[]) obj;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(toJson(arr[i]));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + escapeJson(obj.toString()) + "\"";
    }

    static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }
}
