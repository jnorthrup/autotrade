package com.xtrade;

import org.knowm.xchange.currency.CurrencyPair;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The five major trading pairs supported by the application.
 * Each constant wraps a standard XChange {@link CurrencyPair}.
 */
public enum TradingPair {

    BTC_USD(CurrencyPair.BTC_USD),
    ETH_USD(CurrencyPair.ETH_USD),
    XRP_USD(new CurrencyPair("XRP", "USD")),
    SOL_USD(new CurrencyPair("SOL", "USD")),
    ADA_USD(new CurrencyPair("ADA", "USD"));

    private final CurrencyPair currencyPair;

    private static final List<TradingPair> VALUES =
            Collections.unmodifiableList(Arrays.asList(values()));

    TradingPair(CurrencyPair currencyPair) {
        this.currencyPair = currencyPair;
    }

    /** Returns the XChange {@link CurrencyPair} representation. */
    public CurrencyPair getCurrencyPair() {
        return currencyPair;
    }

    /** Human-readable symbol, e.g. "BTC/USD". */
    public String getSymbol() {
        return currencyPair.base.getCurrencyCode() + "/" + currencyPair.counter.getCurrencyCode();
    }

    @Override
    public String toString() {
        return getSymbol();
    }

    /** Returns an unmodifiable list of all trading pairs. */
    public static List<TradingPair> valuesAsList() {
        return VALUES;
    }
}
