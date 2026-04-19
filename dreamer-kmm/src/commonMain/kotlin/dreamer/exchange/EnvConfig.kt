package dreamer.exchange

import kotlin.time.Duration.Companion.milliseconds

expect fun getenv(name: String): String?

/**
 * Load Robinhood API configuration from environment variables.
 *
 * Required:
 *   - API_KEY
 *
 * Optional:
 *   - PRIVATE_KEY_BASE64 or PRIVATE_KEY
 *   - ROBINHOOD_BASE_URL
 *   - ROBINHOOD_RETRY_DELAY_MS
 *   - ROBINHOOD_REQUEST_TIMEOUT_MS
 *
 * Returns null if API_KEY is missing.
 */
fun loadRobinhoodApiConfigFromEnv(): RobinhoodApiConfig? {
    val apiKey = getenv("API_KEY") ?: return null
    val baseUrl = getenv("ROBINHOOD_BASE_URL") ?: "https://trading.robinhood.com"
    val retryDelayMs = getenv("ROBINHOOD_RETRY_DELAY_MS")?.toLongOrNull() ?: 60_000L
    val requestTimeoutMs = getenv("ROBINHOOD_REQUEST_TIMEOUT_MS")?.toLongOrNull() ?: 20_000L
    return RobinhoodApiConfig(
        apiKey = apiKey,
        baseUrl = baseUrl,
        retryDelay = retryDelayMs.milliseconds,
        requestTimeout = requestTimeoutMs.milliseconds,
    )
}

fun getPrivateKeyBase64FromEnv(): String? = getenv("PRIVATE_KEY_BASE64") ?: getenv("PRIVATE_KEY")

/**
 * Helper to create a RobinhoodClient from env variables.
 * Requires caller to supply an HttpTransport and a signerFactory that
 * can produce a Signer from the private key base64.
 *
 * Returns null if required env vars are not present.
 */
fun createRobinhoodClientFromEnv(http: HttpTransport, signerFactory: (privateKeyBase64: String) -> Signer): RobinhoodClient? {
    val config = loadRobinhoodApiConfigFromEnv() ?: return null
    val priv = getPrivateKeyBase64FromEnv() ?: return null
    val signer = signerFactory(priv)
    return RobinhoodClient(config, http, signer)
}
