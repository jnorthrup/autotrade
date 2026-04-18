package com.xtrade.codec;

import java.util.*;

/**
 * Registry for all 24 codec strategy implementations.
 * Provides lookup by ID, name, and bulk access to the full panel.
 */
public class CodecRegistry {

    private static final Map<Integer, Class<? extends BaseCodecExpert>> registry = new LinkedHashMap<>();
    private static final Map<String, Class<? extends BaseCodecExpert>> nameRegistry = new LinkedHashMap<>();

    static {
        // Register all 24 codecs
        registry.put(1,  Codec01VolatilityBreakout.class);
        registry.put(2,  Codec02MomentumTrend.class);
        registry.put(3,  Codec03MeanReversion.class);
        registry.put(4,  Codec04TrendFollowing.class);
        registry.put(5,  Codec05PairsTrading.class);
        registry.put(6,  Codec06GridTrading.class);
        registry.put(7,  Codec07VolumeProfile.class);
        registry.put(8,  Codec08OrderFlow.class);
        registry.put(9,  Codec09CorrelationTrading.class);
        registry.put(10, Codec10LiquidityMaking.class);
        registry.put(11, Codec11SectorRotation.class);
        registry.put(12, Codec12CompositeAlpha.class);
        registry.put(13, Codec13RsiReversal.class);
        registry.put(14, Codec14BollingerBands.class);
        registry.put(15, Codec15MacdCrossover.class);
        registry.put(16, Codec16StochasticKd.class);
        registry.put(17, Codec17AdxTrendStrength.class);
        registry.put(18, Codec18VwapMeanReversion.class);
        registry.put(19, Codec19KalmanFilterTrend.class);
        registry.put(20, Codec20HurstRegime.class);
        registry.put(21, Codec21RandomForestClassifier.class);
        registry.put(22, Codec22XgboostSignal.class);
        registry.put(23, Codec23TransformerAttention.class);
        registry.put(24, Codec24ZscoreStatArb.class);

        for (Map.Entry<Integer, Class<? extends BaseCodecExpert>> e : registry.entrySet()) {
            try {
                BaseCodecExpert instance = e.getValue().newInstance();
                nameRegistry.put(instance.getStrategyName(), e.getValue());
            } catch (Exception ex) {
                throw new RuntimeException("Failed to instantiate codec " + e.getKey(), ex);
            }
        }
    }

    public static BaseCodecExpert create(int id) {
        Class<? extends BaseCodecExpert> cls = registry.get(id);
        if (cls == null) throw new IllegalArgumentException("Unknown codec ID: " + id);
        try {
            return cls.newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create codec " + id, e);
        }
    }

    public static BaseCodecExpert createByName(String name) {
        Class<? extends BaseCodecExpert> cls = nameRegistry.get(name);
        if (cls == null) throw new IllegalArgumentException("Unknown codec name: " + name);
        try {
            return cls.newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create codec " + name, e);
        }
    }

    public static List<BaseCodecExpert> createAll() {
        List<BaseCodecExpert> panel = new ArrayList<>();
        for (int i = 1; i <= 24; i++) {
            panel.add(create(i));
        }
        return panel;
    }

    public static int registeredCount() {
        return registry.size();
    }

    public static Set<Integer> registeredIds() {
        return Collections.unmodifiableSet(registry.keySet());
    }

    public static Set<String> registeredNames() {
        return Collections.unmodifiableSet(nameRegistry.keySet());
    }

    public static List<Map<String, Object>> listAvailable() {
        List<Map<String, Object>> list = new ArrayList<>();
        for (int i = 1; i <= 24; i++) {
            BaseCodecExpert expert = create(i);
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("id", expert.getCodecId());
            info.put("class_name", expert.getClass().getSimpleName());
            info.put("name", expert.getStrategyName());
            list.add(info);
        }
        return list;
    }
}
