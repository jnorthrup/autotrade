package com.xtrade.showdown;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Replay data source: reads tick data from a CSV file.
 *
 * Expected columns: timestamp, pair, price, volume
 * Rows with the same timestamp are grouped into one tick.
 */
public class ReplayDataSource implements IDataSource {

    private final String filepath;
    private final Integer maxTicks;

    // Pre-parsed groups: each group is a map of pair -> PairTick
    private final List<Map<String, TickData.PairTick>> groups;
    private int tickIdx;

    public ReplayDataSource(String filepath) {
        this(filepath, null);
    }

    public ReplayDataSource(String filepath, Integer maxTicks) {
        this.filepath = filepath;
        this.maxTicks = maxTicks;
        this.groups = new ArrayList<>();
        this.tickIdx = 0;
        load();
    }

    private void load() {
        Path path = Paths.get(filepath);
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("CSV file not found: " + filepath);
        }

        List<Row> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String header = reader.readLine(); // skip header
            if (header == null) return;

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
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV: " + filepath, e);
        }

        // Group by timestamp
        if (rows.isEmpty()) return;

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
    }

    @Override
    public boolean hasNext() {
        if (maxTicks != null && tickIdx >= maxTicks) return false;
        return tickIdx < groups.size();
    }

    @Override
    public TickData next() {
        if (!hasNext()) throw new NoSuchElementException("No more ticks");
        Map<String, TickData.PairTick> group = groups.get(tickIdx);
        tickIdx++;
        return new TickData(group);
    }

    @Override
    public void reset() {
        tickIdx = 0;
    }

    public int totalTicks() {
        return groups.size();
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
}
