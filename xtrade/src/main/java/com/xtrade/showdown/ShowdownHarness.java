package com.xtrade.showdown;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.xtrade.ExchangeService;
import com.xtrade.codec.BaseCodecExpert;
import com.xtrade.codec.CodecFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Multi-agent showdown harness that mirrors the Python ShowdownRunner.
 *
 * Instantiates N CodecStrategy agents, each with its own isolated
 * IndicatorComputer and virtual portfolio. Feeds identical tick data
 * to every agent per cycle, collects their (conviction, direction)
 * signals, translates them to BUY/SELL/HOLD with position sizing,
 * and tracks per-agent AgentMetrics.
 *
 * Supports:
 *   - SimulatedDataSource (random-walk)
 *   - ReplayDataSource (CSV)
 *   - RealtimeDataSource (via ExchangeService)
 *   - JSON output mode (--output json)
 *   - ASCII dashboard mode (--dashboard)
 */
public class ShowdownHarness {

    private static final Logger LOG = LoggerFactory.getLogger(ShowdownHarness.class);

    private final List<Integer> codecIds;
    private final double initialCash;
    private final List<String> pairs;
    private final int numTicks;

    // Per-agent state
    private final Map<String, ShowdownAgent> agents;
    private final Map<String, AgentMetrics> metricsMap;

    private IDataSource dataSource;
    private int tickCount;
    private Map<String, Double> lastPrices;

    // Output mode
    private OutputMode outputMode;
    private String jsonOutputPath;
    private boolean dashboardEnabled;
    private int dashboardRefreshInterval;

    /** Output mode for showdown results. */
    public enum OutputMode {
        TEXT,
        JSON
    }

    /**
     * Construct a ShowdownHarness.
     *
     * @param codecIds    list of codec IDs (1-24) to instantiate
     * @param dataSource  the data source to use (simulated, replay, or realtime)
     * @param initialCash starting paper-cash per agent
     * @param pairs       trading pairs
     * @param numTicks    tick budget for the run
     */
    public ShowdownHarness(
            List<Integer> codecIds,
            IDataSource dataSource,
            double initialCash,
            List<String> pairs,
            int numTicks
    ) {
        this.codecIds = new ArrayList<>(codecIds);
        this.initialCash = initialCash;
        this.pairs = pairs != null ? new ArrayList<>(pairs) : Collections.singletonList("BTC/USDT");
        this.numTicks = numTicks;

        this.agents = new LinkedHashMap<>();
        this.metricsMap = new LinkedHashMap<>();

        for (int cid : this.codecIds) {
            BaseCodecExpert expert = CodecFactory.createExpert(cid);
            ShowdownAgent agent = new ShowdownAgent(expert, initialCash);
            String name = expert.getName();
            agents.put(name, agent);
            metricsMap.put(name, new AgentMetrics(name, initialCash));
        }

        this.dataSource = dataSource;
        this.tickCount = 0;
        this.lastPrices = new LinkedHashMap<>();
        this.outputMode = OutputMode.TEXT;
        this.jsonOutputPath = null;
        this.dashboardEnabled = false;
        this.dashboardRefreshInterval = 5;
    }

    /**
     * Convenience constructor with simulated data source.
     */
    public ShowdownHarness(List<Integer> codecIds, int numTicks) {
        this(codecIds,
                new SimulatedDataSource(Collections.singletonList("BTC/USDT"), numTicks),
                100_000.0,
                Collections.singletonList("BTC/USDT"),
                numTicks);
    }

    // ── Output Configuration ──────────────────────────────────────────

    /**
     * Set the output mode for showdown results.
     *
     * @param mode    TEXT or JSON
     * @param jsonPath file path for JSON output (required when mode is JSON)
     */
    public void setOutputMode(OutputMode mode, String jsonPath) {
        this.outputMode = mode;
        this.jsonOutputPath = jsonPath;
    }

    public OutputMode getOutputMode() { return outputMode; }

    public void setDashboardEnabled(boolean enabled) {
        this.dashboardEnabled = enabled;
    }

    public void setDashboardRefreshInterval(int ticks) {
        this.dashboardRefreshInterval = Math.max(1, ticks);
    }

    /**
     * Run the showdown with the configured data source.
     *
     * @param numTicks override tick limit (null = use constructor value)
     * @return per-agent summary metrics map
     */
    public Map<String, Map<String, Object>> run(Integer numTicks) {
        int limit = numTicks != null ? numTicks : this.numTicks;
        tickCount = 0;

        while (dataSource.hasNext() && tickCount < limit) {
            TickData tickData = dataSource.next();
            double timestamp = System.currentTimeMillis() / 1000.0;

            // Record current prices
            lastPrices.clear();
            for (Map.Entry<String, TickData.PairTick> entry : tickData.getTicks().entrySet()) {
                lastPrices.put(entry.getKey(), entry.getValue().getPrice());
            }

            // Feed identical tick data to every agent
            for (Map.Entry<String, ShowdownAgent> entry : agents.entrySet()) {
                String name = entry.getKey();
                ShowdownAgent agent = entry.getValue();

                List<TradeAction> actions = agent.onTick(tickData);
                metricsMap.get(name).recordTick(
                        agent.getCash(),
                        new HashMap<>(agent.getHoldings()),
                        lastPrices,
                        actions,
                        tickCount,
                        timestamp
                );
            }

            tickCount++;

            // Dashboard refresh
            if (dashboardEnabled && tickCount % dashboardRefreshInterval == 0) {
                printDashboard(tickCount, limit);
            }

            if (tickCount % 10 == 0) {
                LOG.debug("Showdown tick {}/{} processed", tickCount, limit);
            }
        }

        // Handle output modes
        if (outputMode == OutputMode.JSON) {
            writeJsonOutput();
        } else {
            printLeaderboard();
        }

        return getSummary();
    }

    /**
     * Run with default tick limit.
     */
    public Map<String, Map<String, Object>> run() {
        return run(null);
    }

    /**
     * Run a replay from a CSV file.
     */
    public Map<String, Map<String, Object>> runReplay(String filepath, Integer numTicks) {
        ReplayDataSource replay = new ReplayDataSource(filepath, numTicks);
        IDataSource saved = this.dataSource;
        this.dataSource = replay;
        reset();
        Map<String, Map<String, Object>> result = run(numTicks);
        this.dataSource = saved;
        return result;
    }

    // ── Query methods ────────────────────────────────────────────────

    /**
     * Per-agent summary metrics.
     */
    public Map<String, Map<String, Object>> getSummary() {
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        for (Map.Entry<String, AgentMetrics> entry : metricsMap.entrySet()) {
            result.put(entry.getKey(), entry.getValue().computeSummary());
        }
        return result;
    }

    /**
     * Per-agent tick-level snapshots.
     */
    public Map<String, List<Map<String, Object>>> getSnapshots() {
        Map<String, List<Map<String, Object>>> result = new LinkedHashMap<>();
        for (Map.Entry<String, AgentMetrics> entry : metricsMap.entrySet()) {
            result.put(entry.getKey(), entry.getValue().getSnapshots());
        }
        return result;
    }

    /**
     * Per-agent per-tick indicator snapshots.
     *
     * Returns a map: agentName -> list of tick-level snapshots.
     * Each tick-level snapshot maps pair -> indicator map.
     *
     * This is the programmatic API for accessing indicator readings
     * used by each agent at each tick.
     */
    public Map<String, List<Map<String, Map<String, Object>>>> getIndicatorSnapshots() {
        Map<String, List<Map<String, Map<String, Object>>>> result = new LinkedHashMap<>();
        for (Map.Entry<String, ShowdownAgent> entry : agents.entrySet()) {
            result.put(entry.getKey(), entry.getValue().getIndicatorSnapshots());
        }
        return result;
    }

    /**
     * Leaderboard: agents ranked by total PnL descending.
     */
    public List<Map<String, Object>> getLeaderboard() {
        Map<String, Map<String, Object>> summaries = getSummary();
        List<Map<String, Object>> ranked = new ArrayList<>(summaries.values());
        ranked.sort((a, b) -> {
            double pnlA = ((Number) a.get("total_pnl")).doubleValue();
            double pnlB = ((Number) b.get("total_pnl")).doubleValue();
            return Double.compare(pnlB, pnlA); // descending
        });
        for (int i = 0; i < ranked.size(); i++) {
            ranked.get(i).put("rank", i + 1);
        }
        return ranked;
    }

    /**
     * Print a formatted leaderboard table.
     */
    public void printLeaderboard() {
        List<Map<String, Object>> lb = getLeaderboard();
        String hdr = String.format("%4s  %-25s %12s %9s %8s %8s %7s %8s",
                "Rank", "Agent", "P&L", "Ret%", "Sharpe", "HitRate", "Trades", "MaxDD%");
        System.out.println();
        System.out.println("=".repeat(95));
        System.out.println("SHOWDOWN LEADERBOARD");
        System.out.println("=".repeat(95));
        System.out.println(hdr);
        System.out.println("-".repeat(95));
        for (Map<String, Object> e : lb) {
            System.out.printf("%4d  %-25s %12.2f %8.2f%% %8.3f %7.2f%% %7d %7.2f%%%n",
                    e.get("rank"),
                    e.get("agent_name"),
                    ((Number) e.get("total_pnl")).doubleValue(),
                    ((Number) e.get("return_pct")).doubleValue(),
                    ((Number) e.get("sharpe_estimate")).doubleValue(),
                    ((Number) e.get("hit_rate")).doubleValue() * 100.0,
                    ((Number) e.get("trade_count")).intValue(),
                    ((Number) e.get("max_drawdown_pct")).doubleValue() * 100.0);
        }
        System.out.println("=".repeat(95));
        System.out.println();
    }

    // ── JSON Output ──────────────────────────────────────────────────

    /**
     * Build the full showdown results as a structured map suitable for JSON.
     * Contains: leaderboard, equity_curves, indicator_snapshots.
     */
    public Map<String, Object> buildJsonResults() {
        Map<String, Object> results = new LinkedHashMap<>();
        results.put("leaderboard", getLeaderboard());

        // Equity curves per agent
        Map<String, List<Double>> equityCurves = new LinkedHashMap<>();
        for (Map.Entry<String, AgentMetrics> entry : metricsMap.entrySet()) {
            equityCurves.put(entry.getKey(), entry.getValue().getEquityCurve());
        }
        results.put("equity_curves", equityCurves);

        // Per-tick indicator snapshots per agent
        results.put("indicator_snapshots", getIndicatorSnapshots());

        // Per-tick per-agent snapshots (financial)
        results.put("tick_snapshots", getSnapshots());

        // Metadata
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("total_ticks", tickCount);
        metadata.put("initial_cash", initialCash);
        metadata.put("pairs", pairs);
        metadata.put("codec_ids", codecIds);
        metadata.put("timestamp", System.currentTimeMillis() / 1000.0);
        results.put("metadata", metadata);

        return results;
    }

    /**
     * Write full showdown results (leaderboard + equity curves +
     * per-tick indicator snapshots) to a JSON file.
     */
    public void writeJsonOutput() {
        writeJsonOutput(this.jsonOutputPath);
    }

    /**
     * Write full showdown results to a specific JSON file path.
     */
    public void writeJsonOutput(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            filePath = "showdown_results.json";
        }
        Map<String, Object> results = buildJsonResults();
        Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();
        try (FileWriter writer = new FileWriter(filePath)) {
            gson.toJson(results, writer);
            LOG.info("Showdown JSON results written to {}", filePath);
            System.out.println("JSON results written to: " + filePath);
        } catch (IOException e) {
            LOG.error("Failed to write JSON output: {}", e.getMessage());
            System.err.println("ERROR: Failed to write JSON output: " + e.getMessage());
        }
    }

    // ── ASCII Dashboard ──────────────────────────────────────────────

    /**
     * Print a live-updating ASCII dashboard of agent states.
     * Uses ANSI escape codes to refresh in-place.
     */
    public void printDashboard(int currentTick, int totalTicks) {
        // Move cursor up to redraw (skip on first tick)
        int agentCount = agents.size();
        if (currentTick > dashboardRefreshInterval) {
            // Move cursor up: header(3) + agents(agentCount) + footer(1) = agentCount + 4
            System.out.print(String.format("\033[%dA", agentCount + 4));
        }

        // Header
        String progress = String.format("[%d/%d]", currentTick, totalTicks);
        System.out.printf("\033[2K\r  SHOWDOWN DASHBOARD %s  %s%n", progress, "=".repeat(60));
        System.out.printf("\033[2K\r  %-20s %10s %10s %8s %7s %6s%n",
                "Agent", "Equity", "P&L", "Ret%", "Trades", "RSI");
        System.out.printf("\033[2K\r  %s%n", "-".repeat(65));

        // Agent rows
        for (Map.Entry<String, ShowdownAgent> entry : agents.entrySet()) {
            String name = entry.getKey();
            ShowdownAgent agent = entry.getValue();
            AgentMetrics metrics = metricsMap.get(name);

            double equity = metrics.getEquityCurve().isEmpty()
                    ? initialCash
                    : metrics.getEquityCurve().get(metrics.getEquityCurve().size() - 1);
            double pnl = equity - initialCash;
            double retPct = initialCash > 0 ? (pnl / initialCash) * 100.0 : 0.0;
            int trades = metrics.getTotalTrades();

            // Get latest RSI from indicator snapshots
            String rsiStr = "  --";
            List<Map<String, Map<String, Object>>> snapshots = agent.getIndicatorSnapshots();
            if (!snapshots.isEmpty()) {
                Map<String, Map<String, Object>> lastSnap = snapshots.get(snapshots.size() - 1);
                for (Map<String, Object> pairIndicators : lastSnap.values()) {
                    Object rsiObj = pairIndicators.get("rsi");
                    if (rsiObj instanceof Number) {
                        rsiStr = String.format("%5.1f", ((Number) rsiObj).doubleValue());
                        break;
                    }
                }
            }

            String pnlSign = pnl >= 0 ? "+" : "";
            System.out.printf("\033[2K\r  %-20s %10.2f %s%9.2f %7.2f%% %6d %5s%n",
                    name.length() > 20 ? name.substring(0, 20) : name,
                    equity, pnlSign, pnl, retPct, trades, rsiStr);
        }

        System.out.printf("\033[2K\r  %s%n", "=".repeat(65));
        System.out.flush();
    }

    /**
     * Reset every agent and metrics tracker to initial state.
     */
    public void reset() {
        for (ShowdownAgent agent : agents.values()) {
            agent.reset();
        }
        for (String name : metricsMap.keySet()) {
            metricsMap.put(name, new AgentMetrics(name, initialCash));
        }
        tickCount = 0;
        lastPrices.clear();
        dataSource.reset();
    }

    // ── Accessors ────────────────────────────────────────────────────

    public Map<String, ShowdownAgent> getAgents() { return Collections.unmodifiableMap(agents); }
    public Map<String, AgentMetrics> getMetricsMap() { return Collections.unmodifiableMap(metricsMap); }
    public int getTickCount() { return tickCount; }
    public int getNumTicks() { return numTicks; }
}
