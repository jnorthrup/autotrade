package com.xtrade;

import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Immutable application configuration.
 * <p>
 * Reads required environment variables {@code KRAKEN_API_KEY} and
 * {@code KRAKEN_API_SECRET}, the optional {@code KRAKEN_MODE} (sandbox|live,
 * defaults to sandbox), and tuneable parameters from
 * {@code application.properties} on the classpath.
 * <p>
 * Construct via {@link #fromEnv()} or the package-private builder-style
 * constructor used in tests.
 */
public final class AppConfig {

    // ---- Environment variable names ----
    public static final String ENV_API_KEY    = "KRAKEN_API_KEY";
    public static final String ENV_API_SECRET = "KRAKEN_API_SECRET";
    public static final String ENV_MODE       = "KRAKEN_MODE";

    // ---- Properties file keys ----
    static final String PROP_POLL_INTERVAL         = "poll-interval-seconds";
    static final String PROP_INITIAL_VIRTUAL_BALANCE = "initial-virtual-balance";

    // ---- Defaults ----
    static final int    DEFAULT_POLL_INTERVAL    = 60;
    static final String DEFAULT_VIRTUAL_BALANCE  = "10000.00";
    static final Mode   DEFAULT_MODE             = Mode.SANDBOX;

    // ---- Supported trading pairs ----
    private static final List<TradingPair> SUPPORTED_PAIRS =
            Collections.unmodifiableList(java.util.Arrays.asList(TradingPair.values()));

    // ---- Instance state ----
    private final String       apiKey;
    private final String       apiSecret;
    private final Mode         mode;
    private final int          pollIntervalSeconds;
    private final BigDecimal   initialVirtualBalance;

    // ------------------------------------------------------------------ //
    //                        Nested types                                //
    // ------------------------------------------------------------------ //

    /** Operating mode. */
    public enum Mode {
        SANDBOX,
        LIVE;

        /** Case-insensitive parse; returns SANDBOX for null/blank input. */
        public static Mode parse(String value) {
            if (value == null || value.isBlank()) return SANDBOX;
            if ("LIVE".equalsIgnoreCase(value.trim())) return LIVE;
            return SANDBOX;
        }
    }

    /** Thrown when a required environment variable is missing or blank. */
    public static class MissingEnvException extends RuntimeException {
        private final String variableName;

        public MissingEnvException(String message, String variableName) {
            super(message);
            this.variableName = variableName;
        }

        public String getVariableName() {
            return variableName;
        }
    }

    // ------------------------------------------------------------------ //
    //                          Factories                                 //
    // ------------------------------------------------------------------ //

    /**
     * Creates an AppConfig by reading the real environment variables and
     * classpath properties file.
     *
     * @throws MissingEnvException if KRAKEN_API_KEY or KRAKEN_API_SECRET is absent/blank
     */
    public static AppConfig fromEnv() {
        return new AppConfig(
                System.getenv(ENV_API_KEY),
                System.getenv(ENV_API_SECRET),
                System.getenv(ENV_MODE),
                loadProperties()
        );
    }

    /**
     * Package-private constructor for direct injection (used by tests).
     * All parameters may be null except where noted.
     */
    AppConfig(String apiKey, String apiSecret, String modeValue, Properties props) {
        // --- Validate credentials ---
        if (apiKey == null || apiKey.isBlank()) {
            throw new MissingEnvException(
                    "Required environment variable " + ENV_API_KEY + " is not set. "
                    + "Please export your Kraken API key and restart.",
                    ENV_API_KEY);
        }
        if (apiSecret == null || apiSecret.isBlank()) {
            throw new MissingEnvException(
                    "Required environment variable " + ENV_API_SECRET + " is not set. "
                    + "Please export your Kraken API secret and restart.",
                    ENV_API_SECRET);
        }

        this.apiKey    = apiKey.trim();
        this.apiSecret = apiSecret.trim();
        this.mode      = Mode.parse(modeValue);

        // --- Properties with safe defaults ---
        this.pollIntervalSeconds = parseInt(props, PROP_POLL_INTERVAL, DEFAULT_POLL_INTERVAL);
        this.initialVirtualBalance = parseBigDecimal(props, PROP_INITIAL_VIRTUAL_BALANCE,
                new BigDecimal(DEFAULT_VIRTUAL_BALANCE));
    }

    // ------------------------------------------------------------------ //
    //                        Accessors (immutable)                       //
    // ------------------------------------------------------------------ //

    public String       getApiKey()                { return apiKey; }
    public String       getApiSecret()             { return apiSecret; }
    public Mode         getMode()                  { return mode; }
    public int          getPollIntervalSeconds()   { return pollIntervalSeconds; }
    public BigDecimal   getInitialVirtualBalance() { return initialVirtualBalance; }
    public boolean      isLiveMode()               { return mode == Mode.LIVE; }

    /** Returns the fixed list of supported trading pairs. */
    public List<TradingPair> getSupportedPairs()   { return SUPPORTED_PAIRS; }

    // ------------------------------------------------------------------ //
    //                     Properties-file helpers                        //
    // ------------------------------------------------------------------ //

    static Properties loadProperties() {
        Properties props = new Properties();
        try (InputStream in = AppConfig.class.getClassLoader()
                .getResourceAsStream("application.properties")) {
            if (in != null) {
                props.load(in);
            }
        } catch (IOException ignored) {
            // defaults will be used
        }
        return props;
    }

    static int parseInt(Properties props, String key, int defaultVal) {
        if (props == null) return defaultVal;
        String raw = props.getProperty(key);
        if (raw == null || raw.isBlank()) return defaultVal;
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }

    static BigDecimal parseBigDecimal(Properties props, String key, BigDecimal defaultVal) {
        if (props == null) return defaultVal;
        String raw = props.getProperty(key);
        if (raw == null || raw.isBlank()) return defaultVal;
        try {
            return new BigDecimal(raw.trim());
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }
}
