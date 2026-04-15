package com.xtrade;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link AppConfig}.
 * <p>
 * Environment variables are injected directly through the package-private
 * constructor so no OS-level env manipulation or Mockito is needed.
 */
class AppConfigTest {

    private static final String VALID_KEY    = "abc123key";
    private static final String VALID_SECRET = "def456secret";

    // ------------------------------------------------------------------ //
    //                     Happy-path construction                        //
    // ------------------------------------------------------------------ //

    @Test
    void fromEnv_withValidCredentials_succeeds() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, new Properties());

        assertEquals(VALID_KEY,    cfg.getApiKey());
        assertEquals(VALID_SECRET, cfg.getApiSecret());
        assertEquals(AppConfig.Mode.SANDBOX, cfg.getMode());
        assertFalse(cfg.isLiveMode());
    }

    @Test
    void fromEnv_withLiveMode_parsesCorrectly() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, "live", new Properties());
        assertEquals(AppConfig.Mode.LIVE, cfg.getMode());
        assertTrue(cfg.isLiveMode());
    }

    @Test
    void fromEnv_withSandboxModeExplicit_parsesCorrectly() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, "sandbox", new Properties());
        assertEquals(AppConfig.Mode.SANDBOX, cfg.getMode());
    }

    @Test
    void fromEnv_modeIsCaseInsensitive() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, "LiVe", new Properties());
        assertEquals(AppConfig.Mode.LIVE, cfg.getMode());
    }

    @Test
    void fromEnv_blankMode_defaultsToSandbox() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, "  ", new Properties());
        assertEquals(AppConfig.Mode.SANDBOX, cfg.getMode());
    }

    // ------------------------------------------------------------------ //
    //                 Missing-credential validation                      //
    // ------------------------------------------------------------------ //

    @Test
    void nullApiKey_throwsMissingEnvException() {
        AppConfig.MissingEnvException ex = assertThrows(AppConfig.MissingEnvException.class,
                () -> new AppConfig(null, VALID_SECRET, null, new Properties()));

        assertTrue(ex.getMessage().contains(AppConfig.ENV_API_KEY));
        assertEquals(AppConfig.ENV_API_KEY, ex.getVariableName());
    }

    @Test
    void blankApiKey_throwsMissingEnvException() {
        AppConfig.MissingEnvException ex = assertThrows(AppConfig.MissingEnvException.class,
                () -> new AppConfig("   ", VALID_SECRET, null, new Properties()));

        assertTrue(ex.getMessage().contains(AppConfig.ENV_API_KEY));
        assertEquals(AppConfig.ENV_API_KEY, ex.getVariableName());
    }

    @Test
    void emptyApiKey_throwsMissingEnvException() {
        AppConfig.MissingEnvException ex = assertThrows(AppConfig.MissingEnvException.class,
                () -> new AppConfig("", VALID_SECRET, null, new Properties()));

        assertTrue(ex.getMessage().contains(AppConfig.ENV_API_KEY));
    }

    @Test
    void nullApiSecret_throwsMissingEnvException() {
        AppConfig.MissingEnvException ex = assertThrows(AppConfig.MissingEnvException.class,
                () -> new AppConfig(VALID_KEY, null, null, new Properties()));

        assertTrue(ex.getMessage().contains(AppConfig.ENV_API_SECRET));
        assertEquals(AppConfig.ENV_API_SECRET, ex.getVariableName());
    }

    @Test
    void blankApiSecret_throwsMissingEnvException() {
        AppConfig.MissingEnvException ex = assertThrows(AppConfig.MissingEnvException.class,
                () -> new AppConfig(VALID_KEY, "   ", null, new Properties()));

        assertTrue(ex.getMessage().contains(AppConfig.ENV_API_SECRET));
    }

    @Test
    void missingEnvException_storesVariableName() {
        AppConfig.MissingEnvException ex =
                new AppConfig.MissingEnvException("test message", "FOO");
        assertEquals("FOO", ex.getVariableName());
        assertEquals("test message", ex.getMessage());
    }

    // ------------------------------------------------------------------ //
    //                    application.properties loading                  //
    // ------------------------------------------------------------------ //

    @Test
    void defaultPollInterval_whenPropertyMissing() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, new Properties());
        assertEquals(AppConfig.DEFAULT_POLL_INTERVAL, cfg.getPollIntervalSeconds());
    }

    @Test
    void customPollInterval_fromProperties() {
        Properties props = new Properties();
        props.setProperty(AppConfig.PROP_POLL_INTERVAL, "30");
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, props);
        assertEquals(30, cfg.getPollIntervalSeconds());
    }

    @Test
    void invalidPollInterval_usesDefault() {
        Properties props = new Properties();
        props.setProperty(AppConfig.PROP_POLL_INTERVAL, "not-a-number");
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, props);
        assertEquals(AppConfig.DEFAULT_POLL_INTERVAL, cfg.getPollIntervalSeconds());
    }

    @Test
    void defaultVirtualBalance_whenPropertyMissing() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, new Properties());
        assertEquals(new BigDecimal(AppConfig.DEFAULT_VIRTUAL_BALANCE),
                     cfg.getInitialVirtualBalance());
    }

    @Test
    void customVirtualBalance_fromProperties() {
        Properties props = new Properties();
        props.setProperty(AppConfig.PROP_INITIAL_VIRTUAL_BALANCE, "5000.50");
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, props);
        assertEquals(new BigDecimal("5000.50"), cfg.getInitialVirtualBalance());
    }

    @Test
    void invalidVirtualBalance_usesDefault() {
        Properties props = new Properties();
        props.setProperty(AppConfig.PROP_INITIAL_VIRTUAL_BALANCE, "bad-value");
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, props);
        assertEquals(new BigDecimal(AppConfig.DEFAULT_VIRTUAL_BALANCE),
                     cfg.getInitialVirtualBalance());
    }

    @Test
    void nullProperties_usesDefaults() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, null);
        assertEquals(AppConfig.DEFAULT_POLL_INTERVAL, cfg.getPollIntervalSeconds());
        assertEquals(new BigDecimal(AppConfig.DEFAULT_VIRTUAL_BALANCE),
                     cfg.getInitialVirtualBalance());
    }

    // ------------------------------------------------------------------ //
    //                        Properties helpers                          //
    // ------------------------------------------------------------------ //

    @Test
    void parseInt_returnsDefaultOnNullProps() {
        assertEquals(42, AppConfig.parseInt(null, "x", 42));
    }

    @Test
    void parseInt_returnsDefaultOnMissingKey() {
        assertEquals(42, AppConfig.parseInt(new Properties(), "x", 42));
    }

    @Test
    void parseInt_returnsParsedValue() {
        Properties p = new Properties();
        p.setProperty("x", "99");
        assertEquals(99, AppConfig.parseInt(p, "x", 42));
    }

    @Test
    void parseBigDecimal_returnsDefaultOnNullProps() {
        assertEquals(BigDecimal.TEN, AppConfig.parseBigDecimal(null, "x", BigDecimal.TEN));
    }

    // ------------------------------------------------------------------ //
    //                       TradingPair coverage                        //
    // ------------------------------------------------------------------ //

    @Test
    void supportedPairs_returnsAllFivePairs() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, new Properties());
        List<TradingPair> pairs = cfg.getSupportedPairs();
        assertEquals(5, pairs.size());

        // Verify exact order
        assertEquals(TradingPair.BTC_USD, pairs.get(0));
        assertEquals(TradingPair.ETH_USD, pairs.get(1));
        assertEquals(TradingPair.XRP_USD, pairs.get(2));
        assertEquals(TradingPair.SOL_USD, pairs.get(3));
        assertEquals(TradingPair.ADA_USD, pairs.get(4));
    }

    @Test
    void supportedPairs_isUnmodifiable() {
        AppConfig cfg = new AppConfig(VALID_KEY, VALID_SECRET, null, new Properties());
        List<TradingPair> pairs = cfg.getSupportedPairs();
        assertThrows(UnsupportedOperationException.class, () -> pairs.add(TradingPair.BTC_USD));
    }

    // ------------------------------------------------------------------ //
    //                     loadProperties coverage                        //
    // ------------------------------------------------------------------ //

    @Test
    void loadProperties_returnsNonNull() {
        Properties props = AppConfig.loadProperties();
        assertNotNull(props);
        // Should contain values from application.properties on classpath
        assertEquals("60",                  props.getProperty(AppConfig.PROP_POLL_INTERVAL));
        assertEquals("10000.00",            props.getProperty(AppConfig.PROP_INITIAL_VIRTUAL_BALANCE));
    }

    // ------------------------------------------------------------------ //
    //                       Immutability spot-check                      //
    // ------------------------------------------------------------------ //

    @Test
    void constructor_trimsWhitespaceFromCredentials() {
        AppConfig cfg = new AppConfig("  key  ", "  secret  ", null, new Properties());
        assertEquals("key",    cfg.getApiKey());
        assertEquals("secret", cfg.getApiSecret());
    }
}
