package com.xtrade;

/**
 * Trading signal produced by a strategy evaluation.
 */
public enum Signal {
    BUY,
    SELL,
    HOLD;

    @Override
    public String toString() {
        return name();
    }
}
