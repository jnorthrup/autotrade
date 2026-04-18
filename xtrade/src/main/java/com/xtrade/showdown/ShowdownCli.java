package com.xtrade.showdown;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Command-line argument parser for showdown mode.
 *
 * Recognised flags:
 *   --showdown          activate multi-agent showdown mode
 *   --all-codecs        select all 24 codecs
 *   --codecs &lt;spec&gt;     comma/hyphen range spec (e.g. "1,3,5,13-20")
 *   --ticks &lt;N&gt;         number of ticks to simulate (default 100)
 *   --simulated         use simulated (random-walk) data source
 *   --replay &lt;file&gt;     use CSV replay data source
 *   --output &lt;mode&gt;     output mode: "text" (default) or "json" [optional: json:/path/to/file]
 *   --dashboard         enable ASCII dashboard mode during run
 *   (no data flag)      attempt live data source (falls back to simulated)
 */
public class ShowdownCli {

    private static final int TOTAL_CODECS = 24;
    private static final int DEFAULT_TICKS = 100;

    private final boolean showdown;
    private final List<Integer> codecIds;
    private final int ticks;
    private final DataSourceKind dataSourceKind;
    private final String replayFile;
    private final OutputMode outputMode;
    private final String jsonOutputPath;
    private final boolean dashboard;

    private ShowdownCli(Builder builder) {
        this.showdown = builder.showdown;
        this.codecIds = Collections.unmodifiableList(new ArrayList<>(builder.codecIds));
        this.ticks = builder.ticks;
        this.dataSourceKind = builder.dataSourceKind;
        this.replayFile = builder.replayFile;
        this.outputMode = builder.outputMode;
        this.jsonOutputPath = builder.jsonOutputPath;
        this.dashboard = builder.dashboard;
    }

    /** Returns true when --showdown flag is present. */
    public boolean isShowdown() { return showdown; }

    /** Codec IDs to run (1-based). */
    public List<Integer> getCodecIds() { return codecIds; }

    /** Number of ticks. */
    public int getTicks() { return ticks; }

    /** Which data source to use. */
    public DataSourceKind getDataSourceKind() { return dataSourceKind; }

    /** Path to CSV replay file, or null. */
    public String getReplayFile() { return replayFile; }

    /** Output mode for showdown results. */
    public OutputMode getOutputMode() { return outputMode; }

    /** JSON output file path, or null for default. */
    public String getJsonOutputPath() { return jsonOutputPath; }

    /** Whether ASCII dashboard mode is enabled. */
    public boolean isDashboardEnabled() { return dashboard; }

    /** Output mode enum. */
    public enum OutputMode {
        TEXT,
        JSON
    }

    /**
     * Parse raw CLI args into a ShowdownCli instance.
     *
     * @param args raw command-line arguments
     * @return parsed configuration
     * @throws IllegalArgumentException on invalid or conflicting arguments
     */
    public static ShowdownCli parse(String[] args) {
        Builder builder = new Builder();
        if (args == null) return builder.build();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--showdown":
                    builder.showdown = true;
                    break;
                case "--all-codecs":
                    builder.addAllCodecs = true;
                    break;
                case "--codecs":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--codecs requires a value (e.g. 1,3,5,13-20)");
                    }
                    builder.codecsSpec = args[++i];
                    break;
                case "--ticks":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--ticks requires a numeric value");
                    }
                    try {
                        builder.ticks = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("--ticks value must be an integer: " + args[i]);
                    }
                    if (builder.ticks < 1) {
                        throw new IllegalArgumentException("--ticks must be >= 1");
                    }
                    break;
                case "--simulated":
                    builder.dataSourceKind = DataSourceKind.SIMULATED;
                    break;
                case "--replay":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--replay requires a file path");
                    }
                    builder.dataSourceKind = DataSourceKind.REPLAY;
                    builder.replayFile = args[++i];
                    break;
                case "--output":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--output requires a mode (text or json[:path])");
                    }
                    parseOutputSpec(builder, args[++i]);
                    break;
                case "--dashboard":
                    builder.dashboard = true;
                    break;
                // Other flags (like --demo) are handled by Main.createApp; skip them
                default:
                    break;
            }
        }

        return builder.build();
    }

    /**
     * Parse --output spec: "text", "json", or "json:/path/to/file.json".
     */
    private static void parseOutputSpec(Builder builder, String spec) {
        if (spec == null || spec.isEmpty()) {
            return;
        }
        String lower = spec.toLowerCase();
        if (lower.startsWith("json")) {
            builder.outputMode = OutputMode.JSON;
            if (lower.startsWith("json:")) {
                builder.jsonOutputPath = spec.substring(5);
            } else {
                builder.jsonOutputPath = "showdown_results.json";
            }
        } else if (lower.equals("text")) {
            builder.outputMode = OutputMode.TEXT;
        } else {
            throw new IllegalArgumentException("--output mode must be 'text' or 'json' (or 'json:/path')");
        }
    }

    /**
     * Expand a codec specification string into a sorted, unique list of IDs.
     * Supports comma-separated values and hyphen ranges, e.g. "1,3,5,13-20".
     *
     * @param spec the codec specification
     * @return sorted list of unique codec IDs
     */
    static List<Integer> expandCodecSpec(String spec) {
        Set<Integer> ids = new LinkedHashSet<>();
        String[] parts = spec.split(",");
        for (String part : parts) {
            part = part.trim();
            if (part.isEmpty()) continue;
            if (part.contains("-")) {
                String[] range = part.split("-", -1);
                if (range.length != 2) {
                    throw new IllegalArgumentException("Invalid codec range: " + part);
                }
                int start = Integer.parseInt(range[0].trim());
                int end = Integer.parseInt(range[1].trim());
                if (start < 1 || end > TOTAL_CODECS || start > end) {
                    throw new IllegalArgumentException(
                            "Codec range out of bounds [1," + TOTAL_CODECS + "]: " + part);
                }
                for (int id = start; id <= end; id++) {
                    ids.add(id);
                }
            } else {
                int id = Integer.parseInt(part);
                if (id < 1 || id > TOTAL_CODECS) {
                    throw new IllegalArgumentException(
                            "Codec ID out of bounds [1," + TOTAL_CODECS + "]: " + id);
                }
                ids.add(id);
            }
        }
        List<Integer> sorted = new ArrayList<>(ids);
        Collections.sort(sorted);
        return sorted;
    }

    /** Data source types. */
    public enum DataSourceKind {
        SIMULATED,
        REPLAY,
        LIVE
    }

    // ── Builder ────────────────────────────────────────────────────────

    private static class Builder {
        boolean showdown;
        boolean addAllCodecs;
        String codecsSpec;
        int ticks = DEFAULT_TICKS;
        DataSourceKind dataSourceKind = DataSourceKind.SIMULATED;
        String replayFile;
        List<Integer> codecIds = new ArrayList<>();
        OutputMode outputMode = OutputMode.TEXT;
        String jsonOutputPath;
        boolean dashboard;

        ShowdownCli build() {
            // Resolve codec IDs
            if (addAllCodecs && codecsSpec != null) {
                throw new IllegalArgumentException("Cannot specify both --all-codecs and --codecs");
            }
            if (showdown) {
                if (addAllCodecs) {
                    codecIds = new ArrayList<>();
                    for (int i = 1; i <= TOTAL_CODECS; i++) {
                        codecIds.add(i);
                    }
                } else if (codecsSpec != null) {
                    codecIds = expandCodecSpec(codecsSpec);
                } else {
                    // Default: all codecs
                    codecIds = new ArrayList<>();
                    for (int i = 1; i <= TOTAL_CODECS; i++) {
                        codecIds.add(i);
                    }
                }
                if (codecIds.isEmpty()) {
                    throw new IllegalArgumentException("No codec IDs selected for showdown");
                }
                // Validate data source consistency
                if (dataSourceKind == DataSourceKind.REPLAY && replayFile == null) {
                    throw new IllegalArgumentException("--replay requires a file path");
                }
            }
            return new ShowdownCli(this);
        }
    }
}
