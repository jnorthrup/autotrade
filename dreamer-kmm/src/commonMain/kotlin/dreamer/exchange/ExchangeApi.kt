package dreamer.exchange

import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

// ── Platform-injected interfaces ──────────────────────────────────────────────

/**
 * Platform-specific HTTP transport.
 * Implemented via `expect`/`actual` — Ktor on JVM, axios/fetch on JS/WASM.
 */
interface HttpTransport {
    /**
     * Execute an HTTP request and return the response.
     *
     * @param method   HTTP method ("GET", "POST", etc.)
     * @param url      Full URL including query string.
     * @param headers  Map of header name → value.
     * @param body     Request body string, or null for no body.
     * @param timeout  Per-request timeout.
     * @return [HttpResponse] with status code and body text.
     */
    suspend fun request(
        method: String,
        url: String,
        headers: Map<String, String>,
        body: String?,
        timeout: Duration,
    ): HttpResponse
}

/** Immutable HTTP response. */
data class HttpResponse(
    val statusCode: Int,
    val body: String,
)

/**
 * Platform-specific Ed25519 detached signature provider.
 * Implemented via `expect`/`actual` — java.security on JVM, tweetnacl on JS/WASM.
 */
interface Signer {
    /**
     * Produce a detached Ed25519 signature for [message] (UTF-8 bytes).
     * @return Signature bytes.
     */
    fun sign(message: ByteArray): ByteArray
}

// ── Configuration ─────────────────────────────────────────────────────────────

/**
 * Configuration for the Robinhood Crypto Trading API client.
 *
 * @property apiKey         Robinhood API key.
 * @property baseUrl        Base URL (default: "https://trading.robinhood.com").
 * @property retryDelay     Delay between retries on transient failures.
 * @property requestTimeout Per-request timeout.
 */
data class RobinhoodApiConfig(
    val apiKey: String,
    val baseUrl: String = "https://trading.robinhood.com",
    val retryDelay: Duration = 60_000.milliseconds,
    val requestTimeout: Duration = 20_000.milliseconds,
)

// ── Response data classes ─────────────────────────────────────────────────────

/**
 * Account balance snapshot from the Robinhood crypto trading accounts endpoint.
 *
 * Mirrors the parsed fields from `GET /api/v1/crypto/trading/accounts/`.
 */
data class RobinhoodBalance(
    val buyingPower: Double,
    val cashBalance: Double,
    val cryptoBuyingPower: Double,
)

/**
 * Holding record for a single crypto asset.
 *
 * Mirrors items in the `results` array from `GET /api/v1/crypto/trading/holdings/`.
 */
data class RobinhoodHolding(
    val assetCode: String,
    val quantity: Double,
    val costBasis: Double,
)

/**
 * Best bid/ask quote for a single asset pair.
 *
 * Mirrors items from `GET /api/v1/crypto/marketdata/best_bid_ask/`.
 */
data class RobinhoodQuote(
    val symbol: String,
    val bidPrice: Double,
    val askPrice: Double,
    val bidSize: Double,
    val askSize: Double,
)

/**
 * Placed order result from `POST /api/v1/crypto/trading/orders/`.
 */
data class RobinhoodOrder(
    val id: String,
    val symbol: String,
    val side: String,
    val quantity: String,
    val price: String,
    val state: String,
    val createdAt: String,
)

// ── Exchange client interface ─────────────────────────────────────────────────

/**
 * High-level exchange operations used by [dreamer.engine.TradingEngine].
 *
 * This is the interface the engine expects for executing trades and querying
 * portfolio state. [RobinhoodClient] implements it with full retry/validation
 * logic; other exchanges can be swapped in by implementing this interface.
 */
interface ExchangeClient {
    /** Fetch current account balance. */
    suspend fun getBalance(): RobinhoodBalance?

    /** Fetch all crypto holdings. */
    suspend fun getHoldings(): List<RobinhoodHolding>?

    /**
     * Fetch best bid/ask quotes for the given asset codes.
     *
     * @param assetCodes List of base asset codes (e.g. "BTC", "ETH").
     * @return Map of "BASE-USD" → mid-price, or null on graceful failure.
     */
    suspend fun getQuotes(assetCodes: List<String>): Map<String, Double>?

    /**
     * Place a buy order.
     *
     * @param symbol       Trading pair symbol (e.g. "BTC-USD").
     * @param quantityStr  Quantity as a decimal string.
     * @return [RobinhoodOrder] on success, null on graceful failure (duplicate,
     *         insufficient funds, invalid quantity).
     */
    suspend fun placeBuy(symbol: String, quantityStr: String): RobinhoodOrder?

    /**
     * Place a sell order.
     *
     * @param symbol       Trading pair symbol (e.g. "BTC-USD").
     * @param quantityStr  Quantity as a decimal string.
     * @return [RobinhoodOrder] on success, null on graceful failure.
     */
    suspend fun placeSell(symbol: String, quantityStr: String): RobinhoodOrder?
}
