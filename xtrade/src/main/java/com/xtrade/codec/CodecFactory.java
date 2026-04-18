package com.xtrade.codec;

import java.util.ArrayList;
import java.util.List;

/**
 * Factory for creating codec expert strategy instances by ID (1-24).
 *
 * Delegates to CodecRegistry for instantiation but provides the
 * createExpert(id) API used by the ensemble signal pipeline.
 */
public final class CodecFactory {

    private CodecFactory() {
        // utility class
    }

    /**
     * Create a codec expert instance by ID (1-24).
     *
     * @param id codec identifier 1-24
     * @return new instance of the requested codec strategy
     * @throws IllegalArgumentException if id is out of range
     */
    public static BaseCodecExpert createExpert(int id) {
        return CodecRegistry.create(id);
    }

    /**
     * Create fresh instances of all 24 codec experts.
     */
    public static List<BaseCodecExpert> createAllExperts() {
        return CodecRegistry.createAll();
    }

    /**
     * Create fresh instances of the specified codec IDs.
     */
    public static List<BaseCodecExpert> createExperts(int... ids) {
        List<BaseCodecExpert> experts = new ArrayList<>();
        for (int id : ids) {
            experts.add(createExpert(id));
        }
        return experts;
    }
}
