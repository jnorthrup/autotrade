package com.xtrade.codec;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;

/**
 * Tests for CodecRegistry: verifies all 24 codecs are registered
 * and can be instantiated.
 */
public class CodecRegistryTest {

    @Test
    public void testAll24CodecsRegistered() {
        assertEquals(24, CodecRegistry.registeredCount());
    }

    @Test
    public void testCreateEachCodec() {
        for (int i = 1; i <= 24; i++) {
            BaseCodecExpert expert = CodecRegistry.create(i);
            assertNotNull(expert, "Codec " + i + " should not be null");
            assertEquals(i, expert.getCodecId());
        }
    }

    @Test
    public void testCreateAll() {
        List<BaseCodecExpert> panel = CodecRegistry.createAll();
        assertEquals(24, panel.size());
        for (int i = 0; i < 24; i++) {
            assertEquals(i + 1, panel.get(i).getCodecId());
        }
    }

    @Test
    public void testCreateByName() {
        BaseCodecExpert expert = CodecRegistry.createByName("volatility_breakout");
        assertNotNull(expert);
        assertEquals(1, expert.getCodecId());
        assertEquals("volatility_breakout", expert.getStrategyName());
    }

    @Test
    public void testCodecNames() {
        String[] expectedNames = {
            "volatility_breakout", "momentum_trend", "mean_reversion",
            "trend_following", "pairs_trading", "grid_trading",
            "volume_profile", "order_flow", "correlation_trading",
            "liquidity_making", "sector_rotation", "composite_alpha",
            "rsi_reversal", "bollinger_bands", "macd_crossover",
            "stochastic_kd", "adx_trend_strength", "vwap_mean_reversion",
            "kalman_filter_trend", "hurst_regime", "random_forest_classifier",
            "xgboost_signal", "transformer_attention", "zscore_stat_arb"
        };
        for (int i = 0; i < 24; i++) {
            BaseCodecExpert expert = CodecRegistry.create(i + 1);
            assertEquals(expectedNames[i], expert.getStrategyName(),
                    "Codec " + (i + 1) + " name mismatch");
        }
    }

    @Test
    public void testListAvailable() {
        List<Map<String, Object>> list = CodecRegistry.listAvailable();
        assertEquals(24, list.size());
        for (Map<String, Object> info : list) {
            assertTrue(info.containsKey("id"));
            assertTrue(info.containsKey("name"));
            assertTrue(info.containsKey("class_name"));
        }
    }

    @Test
    public void testInvalidIdThrows() {
        assertThrows(IllegalArgumentException.class, () -> CodecRegistry.create(99));
    }

    @Test
    public void testInvalidNameThrows() {
        assertThrows(IllegalArgumentException.class, () -> CodecRegistry.createByName("nonexistent_strategy"));
    }
}
