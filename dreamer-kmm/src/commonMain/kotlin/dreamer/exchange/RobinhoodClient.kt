package dreamer.exchange

import kotlin.math.floor
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

// ── Errors ────────────────────────────────────────────────────────────────────

/**
 * Thrown when a request fails with a non-retryable, non-graceful HTTP status.
 */
class ExchangeException(
    val statusCode: Int,
    val body: String,
    message: String,
) : Exception(message)

// ── Robinhood Crypto Trading API client ───────────────────────────────────────

/**
 * Common-main implementation of the Robinhood Crypto Trading API.
 *
 * All HTTP transport and Ed25519 signing are injected via [HttpTransport] and
 * [Signer] platform interfaces.  The retry loop, time-sync auto-correction,
 * order validation, and graceful 400 handling are kept entirely in common code,
 * mirroring Dreamer 1.2.js `RobinhoodAPI` (lines 1583–1722).
 *
 * @param config   API credentials and timing configuration.
 * @param http     Platform-specific HTTP transport.
 * @param signer   Platform-specific Ed25519 detached signer.
 */
class RobinhoodClient(
    private val config: RobinhoodApiConfig,
    private val http: HttpTransport,
    private val signer: Signer,
    private val minOrderQty: MinOrderQtyMap = emptyMap(),
) : ExchangeClient {

    /** Time offset in milliseconds, auto-corrected on 401 "Timestamp" errors. */
    private var timeOffsetMs: Long = 0L

    /**
     * Wall-clock epoch-millis, injected by platform code via [initClock].
     * Monotonic epoch-ms captured at the same moment is stored in [monoBaseMs].
     */
    private var wallBaseMs: Long = 0L
    private var monoBaseMs: Long = 0L

    // ── Timestamp & signing ───────────────────────────────────────────────

    private fun currentTimestampSeconds(): Long {
        val elapsed = TimeSource.Monotonic.markNow().elapsedNow().inWholeMilliseconds
        val nowMs = wallBaseMs + (elapsed - monoBaseMs) + timeOffsetMs
        return floor(nowMs / 1000.0).toLong()
    }

    /**
     * Set the wall-clock epoch-ms and current monotonic reading.
     * Called from platform `actual` constructors or immediately after construction.
     * Platform code should do: client.initClock(System.currentTimeMillis(), TimeSource.Monotonic.markNow().elapsedNow().inWholeMilliseconds)
     */
    fun initClock(wallClockEpochMs: Long, monotonicMs: Long) {
        wallBaseMs = wallClockEpochMs
        monoBaseMs = monotonicMs
    }

    /** Produce a detached Ed25519 signature over the message string (UTF-8). */
    private fun sign(message: String): String {
        val sigBytes = signer.sign(message.encodeToByteArray())
        return sigBytes.toBase64String()
    }

    /** Encode [ByteArray] to Base64 (commonMain-safe, no java.util.Base64). */
    private fun ByteArray.toBase64String(): String {
        val table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        val sb = StringBuilder(((size + 2) / 3) * 4)
        var i = 0
        while (i < size) {
            val b0 = this[i].toInt() and 0xFF
            val b1 = if (i + 1 < size) this[i + 1].toInt() and 0xFF else 0
            val b2 = if (i + 2 < size) this[i + 2].toInt() and 0xFF else 0
            sb.append(table[b0 ushr 2])
            sb.append(table[((b0 and 0x03) shl 4) or (b1 ushr 4)])
            sb.append(if (i + 1 < size) table[((b1 and 0x0F) shl 2) or (b2 ushr 6)] else '=')
            sb.append(if (i + 2 < size) table[b2 and 0x3F] else '=')
            i += 3
        }
        return sb.toString()
    }

    // ── Core request ──────────────────────────────────────────────────────

    /**
     * Build and sign headers, send the request via [http].
     *
     * @param method   HTTP method.
     * @param path     API path (e.g. "/api/v1/crypto/trading/accounts/").
     * @param bodyObj  Request body object to serialise, or null.
     * @return Raw [HttpResponse].
     */
    private suspend fun request(
        method: String,
        path: String,
        bodyObj: Map<String, Any?>? = null,
    ): HttpResponse {
        val t = currentTimestampSeconds()
        val bodyStr = if (bodyObj != null) serializeJson(bodyObj) else ""
        val toSign = config.apiKey + t.toString() + path + method + bodyStr
        val signature = sign(toSign)

        val headers = mutableMapOf(
            "x-api-key" to config.apiKey,
            "x-signature" to signature,
            "x-timestamp" to t.toString(),
            "Content-Type" to "application/json",
            "Accept" to "application/json",
        )

        return http.request(
            method = method,
            url = config.baseUrl + path,
            headers = headers,
            body = if (bodyObj != null) bodyStr else null,
            timeout = config.requestTimeout,
        )
    }

    // ── Retry loop with error classification ──────────────────────────────

    /**
     * Request with automatic retry, time-sync correction, and graceful handling.
     *
     * Classification (mirrors Dreamer 1.2.js _requestWithRetry):
     *  - 401 + "Timestamp" → auto-correct [timeOffsetMs], retry immediately.
     *  - 400 + duplicate order_id / insufficient funds / invalid quantity → return null (graceful).
     *  - Timeout, network errors, 5xx, 429, 403 → retry after [config.retryDelay].
     *  - Everything else → throw [ExchangeException].
     *
     * @param method   HTTP method.
     * @param path     API path.
     * @param bodyObj  Request body or null.
     * @return Parsed JSON string on success, or null on graceful 400.
     */
    private suspend fun requestWithRetry(
        method: String,
        path: String,
        bodyObj: Map<String, Any?>? = null,
    ): String? {
        while (true) {
            try {
                val resp = request(method, path, bodyObj)
                when {
                    resp.statusCode in 200..299 -> return resp.body
                    resp.statusCode == 401 -> {
                        // Check for timestamp-related 401
                        if (resp.body.contains("Timestamp", ignoreCase = true)) {
                            // Auto-correct: shift time offset by the difference
                            val serverTs = extractTimestampFromBody(resp.body)
                            if (serverTs != null) {
                                val localTs = currentTimestampSeconds()
                                timeOffsetMs += (serverTs - localTs) * 1000L
                            }
                            continue // retry immediately with corrected time
                        }
                        throw ExchangeException(
                            resp.statusCode, resp.body,
                            "Unauthorized (non-timestamp): ${resp.body}"
                        )
                    }
                    resp.statusCode == 400 -> {
                        // Graceful handling of known 400 errors
                        val body = resp.body
                        if (isGraceful400(body)) return null
                        throw ExchangeException(
                            resp.statusCode, body,
                            "Bad request: $body"
                        )
                    }
                    resp.statusCode == 429 || resp.statusCode == 403 -> {
                        // Rate-limited / forbidden — retry after delay
                        delayMs(config.retryDelay.inWholeMilliseconds)
                        continue
                    }
                    resp.statusCode >= 500 -> {
                        // Server error — retry after delay
                        delayMs(config.retryDelay.inWholeMilliseconds)
                        continue
                    }
                    else -> throw ExchangeException(
                        resp.statusCode, resp.body,
                        "Unexpected HTTP ${resp.statusCode}: ${resp.body}"
                    )
                }
            } catch (e: ExchangeException) {
                throw e // re-throw non-retryable ExchangeExceptions
            } catch (e: Exception) {
                // Timeout, network errors — retry after delay
                delayMs(config.retryDelay.inWholeMilliseconds)
                continue
            }
        }
    }

    /** Check if a 400 response body contains a graceful (non-fatal) error. */
    private fun isGraceful400(body: String): Boolean {
        val lower = body.lowercase()
        return lower.contains("duplicate") && lower.contains("order_id")
                || lower.contains("insufficient") && lower.contains("fund")
                || lower.contains("invalid") && lower.contains("quantity")
    }

    /**
     * Attempt to extract a server-provided timestamp from an error body.
     * Looks for a numeric value near the word "timestamp" or a `server_time` field.
     */
    private fun extractTimestampFromBody(body: String): Long? {
        // Try JSON field "server_time" or "timestamp"
        val patterns = listOf(
            Regex(""""?server_time"?\s*:\s*(\d{10,})"""),
            Regex(""""?timestamp"?\s*:\s*(\d{10,})"""),
            Regex("""timestamp[^0-9]*(\d{10,})""", RegexOption.IGNORE_CASE),
        )
        for (p in patterns) {
            val m = p.find(body)
            if (m != null) return m.groupValues[1].toLongOrNull()
        }
        return null
    }

    // ── Platform-safe delay ───────────────────────────────────────────────

    /**
     * Suspend for [ms] milliseconds.
     * Uses kotlinx.coroutines.delay.
     */
    private suspend fun delayMs(ms: Long) {
        kotlinx.coroutines.delay(ms)
    }

    // ── UUID generation (commonMain-safe) ─────────────────────────────────

    /**
     * Generate a random UUID v4 string.
     * Uses [kotlin.random.Random] — no java.util.UUID dependency.
     */
    private fun generateUuid(): String {
        val r = kotlin.random.Random
        val bytes = ByteArray(16) { r.nextInt().toByte() }
        // Set version (4) and variant bits per RFC 4122
        bytes[6] = (bytes[6].toInt() and 0x0F or 0x40).toByte()
        bytes[8] = (bytes[8].toInt() and 0x3F or 0x80).toByte()
        val hexChars = "0123456789abcdef"
        val hex = bytes.joinToString("") { b ->
            val v = b.toInt() and 0xFF
            "${hexChars[v shr 4]}${hexChars[v and 0x0F]}"
        }
        return "${hex.substring(0, 8)}-${hex.substring(8, 12)}-${hex.substring(12, 16)}-${hex.substring(16, 20)}-${hex.substring(20, 32)}"
    }

    // ── JSON serialisation (minimal, commonMain-safe) ─────────────────────

    /** Serialise a simple map to a JSON string. Handles strings, numbers, nulls. */
    private fun serializeJson(obj: Map<String, Any?>): String {
        val sb = StringBuilder()
        sb.append('{')
        var first = true
        for ((k, v) in obj) {
            if (!first) sb.append(',')
            first = false
            sb.append('"').append(k.escapeJson()).append('"').append(':')
            when (v) {
                null -> sb.append("null")
                is String -> sb.append('"').append(v.escapeJson()).append('"')
                is Number -> sb.append(v)
                is Boolean -> sb.append(v)
                is Map<*, *> -> {
                    @Suppress("UNCHECKED_CAST")
                    sb.append(serializeJson(v as Map<String, Any?>))
                }
                else -> sb.append('"').append(v.toString().escapeJson()).append('"')
            }
        }
        sb.append('}')
        return sb.toString()
    }

    private fun String.escapeJson(): String {
        val sb = StringBuilder(length)
        for (c in this) when (c) {
            '"' -> sb.append("\\\"")
            '\\' -> sb.append("\\\\")
            '\n' -> sb.append("\\n")
            '\r' -> sb.append("\\r")
            '\t' -> sb.append("\\t")
            else -> sb.append(c)
        }
        return sb.toString()
    }

    // ── JSON parsing helpers (minimal, commonMain-safe) ───────────────────

    /**
     * Extract a top-level string value from a JSON body.
     * Returns null if key is not found.
     */
    private fun jsonGetString(body: String, key: String): String? {
        val pattern = Regex(""""?$key"?\s*:\s*"([^"]*)"""")
        return pattern.find(body)?.groupValues?.get(1)
    }

    /** Extract a top-level numeric value from a JSON body. */
    private fun jsonGetDouble(body: String, key: String): Double? {
        val pattern = Regex(""""?$key"?\s*:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)""")
        return pattern.find(body)?.groupValues?.get(1)?.toDoubleOrNull()
    }

    // ── Order validation ──────────────────────────────────────────────────

    /**
     * Validate order parameters before submission.
     *
     * Mirrors Dreamer 1.2.js `_validateOrderParams`:
     *  - Symbol must contain "-USD".
     *  - Quantity (as a double) must be > 0.
     *  - Quantity must exceed the minimum order quantity for the symbol
     *    (via [checkMinQuantity] from ExchangeMath.kt).
     *
     * @throws IllegalArgumentException if validation fails.
     */
    private fun validateOrderParams(symbol: String, quantityStr: String, side: String) {
        require("-USD" in symbol) {
            "Symbol must contain '-USD', got: $symbol"
        }
        val qty = quantityStr.toDoubleOrNull()
        require(qty != null && qty > 0.0) {
            "Quantity must be a positive number, got: $quantityStr"
        }
        require(side == "buy" || side == "sell") {
            "Side must be 'buy' or 'sell', got: $side"
        }
        require(checkMinQuantity(symbol, qty, minOrderQty)) {
            "Quantity $qty is below minimum order quantity for $symbol"
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Fetch account balance.
     *
     * `GET /api/v1/crypto/trading/accounts/`
     * Parses `buying_power`, `cash_balance`, `crypto_buying_power`.
     */
    override suspend fun getBalance(): RobinhoodBalance? {
        val body = requestWithRetry("GET", "/api/v1/crypto/trading/accounts/") ?: return null

        // JS: Array.isArray(data.results) ? data.results[0] : data
        // Find the first account object — look for a block containing "buying_power" or "cash_balance"
        val accountBlock = extractFirstAccount(body) ?: body

        val buyingPower = jsonGetDouble(accountBlock, "buying_power") ?: 0.0
        val cashBalance = jsonGetDouble(accountBlock, "cash_balance") ?: 0.0
        val cryptoBuyingPower = jsonGetDouble(accountBlock, "crypto_buying_power") ?: 0.0

        // JS returns first non-null field as a scalar balance, but we return all fields
        return RobinhoodBalance(
            buyingPower = if (!buyingPower.isNaN()) buyingPower else 0.0,
            cashBalance = if (!cashBalance.isNaN()) cashBalance else 0.0,
            cryptoBuyingPower = if (!cryptoBuyingPower.isNaN()) cryptoBuyingPower else 0.0,
        )
    }

    /** Extract the first account object from a potential results array. */
    private fun extractFirstAccount(body: String): String? {
        // Try to find "results":[{...}] and extract first object
        val arrStart = body.indexOf("\"results\"")
        if (arrStart < 0) return null
        val objStart = body.indexOf('{', arrStart + 10)
        if (objStart < 0) return null
        var depth = 0
        var i = objStart
        while (i < body.length) {
            when (body[i]) {
                '{' -> depth++
                '}' -> { depth--; if (depth == 0) return body.substring(objStart, i + 1) }
            }
            i++
        }
        return null
    }

    /**
     * Fetch all crypto holdings.
     *
     * `GET /api/v1/crypto/trading/holdings/`
     * Returns the `results` array items.
     */
    override suspend fun getHoldings(): List<RobinhoodHolding>? {
        val body = requestWithRetry("GET", "/api/v1/crypto/trading/holdings/") ?: return null

        // Minimal JSON array parsing — split on objects
        val results = mutableListOf<RobinhoodHolding>()
        val objPattern = Regex("""\{[^{}]*"asset_code"\s*:\s*"([^"]*)"[^{}]*\}""")
        for (m in objPattern.findAll(body)) {
            val obj = m.value
            val code = jsonGetString(obj, "asset_code") ?: continue
            val qty = jsonGetDouble(obj, "quantity") ?: 0.0
            val cost = jsonGetDouble(obj, "cost_basis") ?: 0.0
            results.add(RobinhoodHolding(assetCode = code, quantity = qty, costBasis = cost))
        }
        return results
    }

    /**
     * Fetch best bid/ask quotes for the given asset codes.
     *
     * `GET /api/v1/crypto/marketdata/best_bid_ask/?symbols=SYMBOL1,SYMBOL2`
     * Returns a map of "BASE-USD" → mid-price.
     */
    override suspend fun getQuotes(assetCodes: List<String>): Map<String, Double>? {
        if (assetCodes.isEmpty()) return emptyMap()
        val body = requestWithRetry(
            "GET",
            "/api/v1/crypto/marketdata/best_bid_ask/"
        ) ?: return null

        // Parse all quote objects from results array
        // JS: for (const quote of quotes) { sym = quote.symbol.replace("-USD",""); ... }
        val codesSet = assetCodes.toSet()
        val result = mutableMapOf<String, Double>()
        val objPattern = Regex("""\{[^{}]*"symbol"\s*:\s*"([^"]*)"[^{}]*\}""")
        for (m in objPattern.findAll(body)) {
            val obj = m.value
            val rawSymbol = jsonGetString(obj, "symbol") ?: continue
            val sym = rawSymbol.replace("-USD", "")
            if (sym !in codesSet) continue
            val rawPrice = jsonGetDouble(obj, "price")
            if (rawPrice != null && !rawPrice.isNaN() && rawPrice > 0.0) {
                result[sym] = roundTo10(rawPrice)
            }
        }
        return result
    }

    /** Round to 10 decimal places — mirrors JS Number(rawPrice.toFixed(10)) */
    private fun roundTo10(v: Double): Double {
        val factor = 10_000_000_000.0
        return kotlin.math.round(v * factor) / factor
    }

    /**
     * Place a buy order.
     *
     * `POST /api/v1/crypto/trading/orders/`
     * Validates parameters via [validateOrderParams] before submission.
     *
     * @return [RobinhoodOrder] on success, null on graceful 400 failure.
     */
    override suspend fun placeBuy(symbol: String, quantityStr: String): RobinhoodOrder? {
        validateOrderParams(symbol, quantityStr, "buy")
        return placeOrder(symbol, quantityStr, "buy")
    }

    /**
     * Place a sell order.
     *
     * `POST /api/v1/crypto/trading/orders/`
     * Validates parameters via [validateOrderParams] before submission.
     *
     * @return [RobinhoodOrder] on success, null on graceful 400 failure.
     */
    override suspend fun placeSell(symbol: String, quantityStr: String): RobinhoodOrder? {
        validateOrderParams(symbol, quantityStr, "sell")
        return placeOrder(symbol, quantityStr, "sell")
    }

    /**
     * Internal order submission.
     *
     * Mirrors Dreamer 1.2.js placeSell/placeBuy:
     * - client_order_id: UUID for idempotency
     * - side: "buy" or "sell"
     * - type: "market"
     * - market_order_config: { asset_quantity: quantityStr }
     */
    private suspend fun placeOrder(
        symbol: String,
        quantityStr: String,
        side: String,
    ): RobinhoodOrder? {
        val orderId = generateUuid()
        val bodyObj = mapOf<String, Any?>(
            "client_order_id" to orderId,
            "side" to side,
            "type" to "market",
            "symbol" to symbol,
            "market_order_config" to mapOf(
                "asset_quantity" to quantityStr
            ),
        )
        val responseBody = requestWithRetry(
            "POST",
            "/api/v1/crypto/trading/orders/",
            bodyObj,
        ) ?: return null

        return RobinhoodOrder(
            id = jsonGetString(responseBody, "id") ?: "",
            symbol = jsonGetString(responseBody, "symbol") ?: symbol,
            side = jsonGetString(responseBody, "side") ?: side,
            quantity = jsonGetString(responseBody, "quantity") ?: quantityStr,
            price = jsonGetString(responseBody, "price") ?: "",
            state = jsonGetString(responseBody, "state") ?: "",
            createdAt = jsonGetString(responseBody, "created_at") ?: "",
        )
    }
}
