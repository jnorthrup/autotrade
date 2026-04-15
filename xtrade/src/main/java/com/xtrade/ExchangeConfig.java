package com.xtrade;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.kraken.KrakenExchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.service.marketdata.MarketDataService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Reads Kraken API credentials from environment variables, builds a KrakenExchange
 * instance via ExchangeFactory, and validates connectivity. Supports a paper/dry-run
 * mode flag that prevents live order submission.
 */
public class ExchangeConfig {

    private static final Logger LOG = LoggerFactory.getLogger(ExchangeConfig.class);

    static final String ENV_API_KEY = "KRAKEN_API_KEY";
    static final String ENV_SECRET_KEY = "KRAKEN_SECRET_KEY";
    static final String ENV_PAPER_MODE = "PAPER_MODE";

    private final String apiKey;
    private final String secretKey;
    private final boolean paperMode;
    private final Exchange exchange;

    /**
     * Exception thrown when required environment variables are missing.
     */
    public static class MissingCredentialException extends RuntimeException {
        private final String missingVariable;

        public MissingCredentialException(String message, String missingVariable) {
            super(message);
            this.missingVariable = missingVariable;
        }

        public String getMissingVariable() {
            return missingVariable;
        }
    }

    /**
     * Constructs an ExchangeConfig by reading credentials and the paper-mode flag
     * from environment variables. If either credential is missing, throws
     * MissingCredentialException (the caller should log and exit).
     */
    public ExchangeConfig() {
        this(System.getenv(ENV_API_KEY), System.getenv(ENV_SECRET_KEY), resolvePaperMode());
    }

    /**
     * Package-private constructor for testing — allows injecting credentials and
     * paper-mode without touching the real environment.
     */
    ExchangeConfig(String apiKey, String secretKey, boolean paperMode) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new MissingCredentialException(
                    "Missing required environment variable: " + ENV_API_KEY + ". Set your Kraken API key and restart.",
                    ENV_API_KEY);
        }
        if (secretKey == null || secretKey.isBlank()) {
            throw new MissingCredentialException(
                    "Missing required environment variable: " + ENV_SECRET_KEY + ". Set your Kraken secret key and restart.",
                    ENV_SECRET_KEY);
        }

        this.apiKey = apiKey;
        this.secretKey = secretKey;
        this.paperMode = paperMode;

        ExchangeSpecification spec = new KrakenExchange().getDefaultExchangeSpecification();
        spec.setApiKey(this.apiKey);
        spec.setSecretKey(this.secretKey);

        this.exchange = ExchangeFactory.INSTANCE.createExchange(spec);
        LOG.info("KrakenExchange instance created successfully (paperMode={})", paperMode);
    }

    /**
     * Reads the PAPER_MODE environment variable (or system property). Defaults to
     * {@code true} so that the application is safe by default.
     */
    static boolean resolvePaperMode() {
        String envVal = System.getenv(ENV_PAPER_MODE);
        if (envVal != null) {
            return Boolean.parseBoolean(envVal.trim());
        }
        String propVal = System.getProperty(ENV_PAPER_MODE);
        if (propVal != null) {
            return Boolean.parseBoolean(propVal.trim());
        }
        return true; // safe default
    }

    /**
     * Validates connectivity by fetching ticker data for BTC/USD from the Kraken API.
     *
     * @return true if the API responded without error
     */
    public boolean validateConnectivity() {
        try {
            MarketDataService marketDataService = exchange.getMarketDataService();
            marketDataService.getTicker(CurrencyPair.BTC_USD);
            LOG.info("Kraken connectivity validated – BTC/USD ticker retrieved successfully.");
            return true;
        } catch (IOException e) {
            LOG.error("Failed to validate Kraken connectivity: {}", e.getMessage(), e);
            return false;
        }
    }

    // ---- accessors ----

    public String getApiKey() {
        return apiKey;
    }

    public String getSecretKey() {
        return secretKey;
    }

    public boolean isPaperMode() {
        return paperMode;
    }

    public Exchange getExchange() {
        return exchange;
    }
}
