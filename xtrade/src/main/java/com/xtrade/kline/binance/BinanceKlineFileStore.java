package com.xtrade.kline.binance;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class BinanceKlineFileStore {
    private BinanceKlineFileStore() {
    }

    static Map<Long, BinanceKlineRecord> read(Path csvPath) throws IOException {
        Map<Long, BinanceKlineRecord> rows = new LinkedHashMap<>();
        if (!Files.exists(csvPath)) {
            return rows;
        }
        for (String line : Files.readAllLines(csvPath, StandardCharsets.UTF_8)) {
            BinanceKlineRecord record = BinanceKlineRecord.parseCsv(line);
            if (record != null) {
                rows.put(record.openTime(), record);
            }
        }
        return rows;
    }

    static void write(Path csvPath, Map<Long, BinanceKlineRecord> rows) throws IOException {
        Files.createDirectories(csvPath.getParent());
        List<String> lines = new ArrayList<>();
        lines.add(BinanceKlineRecord.CSV_HEADER);
        rows.values().stream()
                .sorted((left, right) -> Long.compare(left.openTime(), right.openTime()))
                .map(BinanceKlineRecord::toCsvLine)
                .forEach(lines::add);
        Files.write(csvPath, lines, StandardCharsets.UTF_8);
    }

    static long lastCloseTime(Path csvPath) throws IOException {
        long last = -1L;
        for (BinanceKlineRecord record : read(csvPath).values()) {
            last = Math.max(last, record.closeTime());
        }
        return last;
    }
}
