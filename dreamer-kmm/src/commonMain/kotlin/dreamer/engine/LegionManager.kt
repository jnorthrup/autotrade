package dreamer.engine

// ---------------------------------------------------------------------------
// LegionManager — pure-data port of Dreamer 1.2.js lines 89-249
// No EventEmitter, no async, no network I/O.
// ---------------------------------------------------------------------------

// ---------- enumerations & data classes ----------

enum class AssetHeat { COLD, WARM, HOT, INFERNO }

data class LegionConfig(
    val totalShadowCapacity: Int = 50,
    val passiveAssetMonitorCount: Int = 5,
    val activeAssetSwarmDensity: Int = 10
)

data class ManagerConfig(
    val heatCheckIntervalMs: Long = 60_000L,
    val cullThresholdHours: Double = 48.0,
    val staleGenomeHours: Double = 72.0
)

data class ShadowInstance(
    val assignedAsset: String,
    val startTime: Long,
    val lastTradeTime: Long,
    val killMe: Boolean = false
)

data class LegionState(
    val shadows: List<ShadowInstance>,
    val heatMap: Map<String, AssetHeat>,
    val lastHeatCheck: Long,
    val lastOptimizationRequest: Map<String, Long>
)

// ---------- heat thresholds ----------
private const val INFERNO_THRESHOLD = 0.005
private const val HOT_THRESHOLD = 0.02
private const val WARM_THRESHOLD = 0.05

// ---------- pure functions ----------

/**
 * Compute heat map for every asset.
 *
 * deviation = |value - baseline| / baseline  (baseline <= 0 → COLD)
 * triggerDist = min(|deviation - harvestTrigger|, |deviation - rebalanceTrigger|)
 *
 * INFERNO < 0.005, HOT < 0.02, WARM < 0.05, else COLD
 *
 * Port of JS lines 109-131.
 */
fun updateHeatMap(
    baselines: Map<String, Double>,
    assets: List<Pair<String, Double>>,
    harvestTrigger: Double,
    rebalanceTrigger: Double
): Map<String, AssetHeat> {
    val result = mutableMapOf<String, AssetHeat>()
    for ((sym, value) in assets) {
        val baseline = baselines[sym] ?: 0.0
        if (baseline <= 0.0) {
            result[sym] = AssetHeat.COLD
            continue
        }
        val deviation = kotlin.math.abs((value - baseline) / baseline)
        val triggerDist = kotlin.math.min(
            kotlin.math.abs(deviation - harvestTrigger),
            kotlin.math.abs(deviation - rebalanceTrigger)
        )
        result[sym] = when {
            triggerDist < INFERNO_THRESHOLD -> AssetHeat.INFERNO
            triggerDist < HOT_THRESHOLD     -> AssetHeat.HOT
            triggerDist < WARM_THRESHOLD    -> AssetHeat.WARM
            else                            -> AssetHeat.COLD
        }
    }
    return result
}

/**
 * Return surviving shadows after culling.
 *
 * A shadow is culled when any of:
 *   killMe == true
 *   idleHours > cullThresholdHours
 *   ageHours > staleGenomeHours
 *   capacity > 90 % AND asset is COLD AND brothers > passiveAssetMonitorCount
 *
 * Port of JS lines 133-157.
 */
fun cullShadows(
    shadows: List<ShadowInstance>,
    heatMap: Map<String, AssetHeat>,
    nowMs: Long,
    config: LegionConfig,
    managerConfig: ManagerConfig
): List<ShadowInstance> {
    // Pre-compute brother counts per asset
    val brotherCount = shadows.groupingBy { it.assignedAsset }.eachCount()
    val capacityOver90 = shadows.size > config.totalShadowCapacity * 0.9

    return shadows.filterNot { shadow ->
        val asset = shadow.assignedAsset
        val heat = heatMap[asset] ?: AssetHeat.COLD

        // Self-termination flag
        if (shadow.killMe) return@filterNot true

        // Idle too long
        val idleMs = nowMs - (shadow.lastTradeTime.takeIf { it > 0L } ?: shadow.startTime)
        val idleHours = idleMs / 3_600_000.0
        if (idleHours > managerConfig.cullThresholdHours) return@filterNot true

        // Stale genome
        val ageHours = (nowMs - shadow.startTime) / 3_600_000.0
        if (ageHours > managerConfig.staleGenomeHours) return@filterNot true

        // Over-capacity + cold + too many monitors for same asset
        if (capacityOver90 && heat == AssetHeat.COLD) {
            val brothers = brotherCount[asset] ?: 0
            if (brothers > config.passiveAssetMonitorCount) return@filterNot true
        }

        false // keep
    }
}

/**
 * Compute list of asset names that need additional shadows spawned.
 *
 * For each HOT / INFERNO asset, if current shadow count < activeAssetSwarmDensity,
 * emit the deficit count of asset names.
 *
 * Port of JS lines 159-168.
 */
fun spawnShadows(
    shadows: List<ShadowInstance>,
    heatMap: Map<String, AssetHeat>,
    config: LegionConfig
): List<String> {
    val currentCount = shadows.groupBy { it.assignedAsset }
        .mapValues { (_, list) -> list.size }

    val toSpawn = mutableListOf<String>()
    for ((asset, heat) in heatMap) {
        if (heat == AssetHeat.HOT || heat == AssetHeat.INFERNO) {
            val count = currentCount[asset] ?: 0
            val deficit = config.activeAssetSwarmDensity - count
            repeat(deficit.coerceAtLeast(0)) {
                toSpawn.add(asset)
            }
        }
    }
    return toSpawn
}

/**
 * Single tick of the LegionManager heartbeat.
 *
 * If not enough time has elapsed since lastHeatCheck, returns state unchanged.
 * Otherwise: updateHeatMap → cullShadows → compute spawn list → new LegionState.
 *
 * Port of JS lines 101-108.
 */
fun heartbeat(
    state: LegionState,
    baselines: Map<String, Double>,
    assets: List<Pair<String, Double>>,
    harvestTrigger: Double,
    rebalanceTrigger: Double,
    nowMs: Long,
    config: LegionConfig,
    managerConfig: ManagerConfig
): LegionState {
    if (nowMs - state.lastHeatCheck < managerConfig.heatCheckIntervalMs) {
        return state
    }

    val newHeatMap = updateHeatMap(baselines, assets, harvestTrigger, rebalanceTrigger)
    val survivingShadows = cullShadows(state.shadows, newHeatMap, nowMs, config, managerConfig)
    // spawnShadows result is returned to the caller for actual deployment;
    // shadows list in state reflects only the surviving set.
    // The caller is responsible for creating ShadowInstances for each spawn name
    // and appending them to the shadows list before the next heartbeat.

    return state.copy(
        shadows = survivingShadows,
        heatMap = newHeatMap,
        lastHeatCheck = nowMs
    )
}
