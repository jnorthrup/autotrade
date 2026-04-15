package com.xtrade;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.service.marketdata.MarketDataService;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Date;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MarketDataServiceImplTest {

    @Mock
    private Exchange exchange;

    @Mock
    private MarketDataService marketDataService;

    private MarketDataServiceImpl service;

    @BeforeEach
    void setUp() {
        when(exchange.getMarketDataService()).thenReturn(marketDataService);
        service = new MarketDataServiceImpl(exchange, 5L);
    }

    @AfterEach
    void tearDown() {
        if (service.isRunning()) {
            service.stopPolling();
        }
    }

    // ---- Constructor validation ----

    @Test
    void constructor_rejectsNullExchange() {
        assertThrows(NullPointerException.class, () -> new MarketDataServiceImpl(null));
    }

    @Test
    void constructor_rejectsZeroInterval() {
        assertThrows(IllegalArgumentException.class, () -> new MarketDataServiceImpl(exchange, 0));
    }

    @Test
    void constructor_rejectsNegativeInterval() {
        assertThrows(IllegalArgumentException.class, () -> new MarketDataServiceImpl(exchange, -1));
    }

    @Test
    void constructor_defaultIntervalIs60Seconds() {
        MarketDataServiceImpl svc = new MarketDataServiceImpl(exchange);
        assertEquals(60L, svc.getPollIntervalSeconds());
    }

    @Test
    void constructor_customIntervalIsStored() {
        assertEquals(5L, service.getPollIntervalSeconds());
    }

    // ---- Monitored pairs ----

    @Test
    void monitoredPairs_containsAllFive() {
        var pairs = service.getMonitoredPairs();
        assertEquals(5, pairs.size());
        assertTrue(pairs.contains(CurrencyPair.BTC_USD));
        assertTrue(pairs.contains(CurrencyPair.ETH_USD));
        assertTrue(pairs.contains(CurrencyPair.XRP_USD));
        assertTrue(pairs.contains(new CurrencyPair("SOL/USD")));
        assertTrue(pairs.contains(CurrencyPair.ADA_USD));
    }

    @Test
    void monitoredPairs_staticConstantMatches() {
        assertEquals(MarketDataServiceImpl.MONITORED_PAIRS, service.getMonitoredPairs());
    }

    // ---- fetchAllTickers success ----

    @Test
    void fetchAllTickers_returnsAllPairs() throws Exception {
        // Stub getTicker to return a valid ticker for any pair
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertEquals(5, result.size());
        verify(marketDataService, times(5)).getTicker(any(CurrencyPair.class));
    }

    @Test
    void fetchAllTickers_returnsCorrectLastPrice() throws Exception {
        Ticker btcTicker = buildTicker(CurrencyPair.BTC_USD, "85000.50", "85000.25", "85000.75", "1234.56");

        // Use lenient for the general stub, specific one for BTC_USD
        lenient().when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));
        when(marketDataService.getTicker(CurrencyPair.BTC_USD)).thenReturn(btcTicker);

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertNotNull(result.get(CurrencyPair.BTC_USD));
        assertEquals(new BigDecimal("85000.50"), result.get(CurrencyPair.BTC_USD).getLast());
        assertEquals(new BigDecimal("85000.25"), result.get(CurrencyPair.BTC_USD).getBid());
        assertEquals(new BigDecimal("85000.75"), result.get(CurrencyPair.BTC_USD).getAsk());
        assertEquals(new BigDecimal("1234.56"), result.get(CurrencyPair.BTC_USD).getVolume());
    }

    @Test
    void fetchAllTickers_storesLatestTickers() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        service.fetchAllTickers();

        Map<CurrencyPair, Ticker> latest = service.getLatestTickers();
        assertNotNull(latest);
        assertEquals(5, latest.size());
    }

    @Test
    void fetchAllTickers_resultIsUnmodifiable() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertThrows(UnsupportedOperationException.class, () -> result.put(CurrencyPair.BTC_USD, null));
    }

    // ---- fetchAllTickers partial failure ----

    @Test
    void fetchAllTickers_continuesOnSinglePairFailure() throws Exception {
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("API timeout"));
        when(marketDataService.getTicker(CurrencyPair.ETH_USD))
                .thenReturn(buildTicker(CurrencyPair.ETH_USD));
        when(marketDataService.getTicker(CurrencyPair.XRP_USD))
                .thenReturn(buildTicker(CurrencyPair.XRP_USD));
        when(marketDataService.getTicker(new CurrencyPair("SOL/USD")))
                .thenReturn(buildTicker(new CurrencyPair("SOL/USD")));
        when(marketDataService.getTicker(CurrencyPair.ADA_USD))
                .thenReturn(buildTicker(CurrencyPair.ADA_USD));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertEquals(4, result.size());
        assertFalse(result.containsKey(CurrencyPair.BTC_USD));
        assertTrue(result.containsKey(CurrencyPair.ETH_USD));
    }

    @Test
    void fetchAllTickers_handlesAllFailures() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenThrow(new IOException("Service unavailable"));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertTrue(result.isEmpty());
    }

    @Test
    void fetchAllTickers_handlesRuntimeException() throws Exception {
        when(marketDataService.getTicker(CurrencyPair.ETH_USD))
                .thenThrow(new RuntimeException("Unexpected error"));
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenReturn(buildTicker(CurrencyPair.BTC_USD));
        when(marketDataService.getTicker(CurrencyPair.XRP_USD))
                .thenReturn(buildTicker(CurrencyPair.XRP_USD));
        when(marketDataService.getTicker(new CurrencyPair("SOL/USD")))
                .thenReturn(buildTicker(new CurrencyPair("SOL/USD")));
        when(marketDataService.getTicker(CurrencyPair.ADA_USD))
                .thenReturn(buildTicker(CurrencyPair.ADA_USD));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertEquals(4, result.size());
        assertFalse(result.containsKey(CurrencyPair.ETH_USD));
    }

    @Test
    void fetchAllTickers_handlesNullTickerReturn() throws Exception {
        CurrencyPair solUsd = new CurrencyPair("SOL/USD");
        when(marketDataService.getTicker(solUsd)).thenReturn(null);
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenReturn(buildTicker(CurrencyPair.BTC_USD));
        when(marketDataService.getTicker(CurrencyPair.ETH_USD))
                .thenReturn(buildTicker(CurrencyPair.ETH_USD));
        when(marketDataService.getTicker(CurrencyPair.XRP_USD))
                .thenReturn(buildTicker(CurrencyPair.XRP_USD));
        when(marketDataService.getTicker(CurrencyPair.ADA_USD))
                .thenReturn(buildTicker(CurrencyPair.ADA_USD));

        Map<CurrencyPair, Ticker> result = service.fetchAllTickers();

        assertEquals(4, result.size());
        assertFalse(result.containsKey(solUsd));
    }

    // ---- Polling ----

    @Test
    void startPolling_setsRunningToTrue() {
        assertFalse(service.isRunning());
        service.startPolling();
        assertTrue(service.isRunning());
    }

    @Test
    void startPolling_idempotentDoesNotThrow() {
        service.startPolling();
        assertDoesNotThrow(() -> service.startPolling());
        assertTrue(service.isRunning());
    }

    @Test
    void stopPolling_setsRunningToFalse() {
        service.startPolling();
        assertTrue(service.isRunning());
        service.stopPolling();
        assertFalse(service.isRunning());
    }

    @Test
    void stopPolling_whenNotRunning_isNoOp() {
        assertFalse(service.isRunning());
        assertDoesNotThrow(() -> service.stopPolling());
        assertFalse(service.isRunning());
    }

    @Test
    void polling_triggersFetchAtLeastOnce() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        service.startPolling();
        Thread.sleep(500); // Give scheduler time to fire

        assertTrue(service.getLatestTickers().size() >= 1,
                "Expected at least one ticker after polling starts");
    }

    @Test
    void polling_continuesAfterFailure() throws Exception {
        // First call throws, subsequent calls succeed
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenThrow(new IOException("Transient failure"))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        service.startPolling();
        Thread.sleep(500);

        // Service should still be running despite the failure
        assertTrue(service.isRunning());
    }

    // ---- logTicker does not throw ----

    @Test
    void logTicker_doesNotThrowWithNullFields() {
        Ticker ticker = buildTicker(CurrencyPair.BTC_USD, null, null, null, null);
        assertDoesNotThrow(() -> service.logTicker(CurrencyPair.BTC_USD, ticker));
    }

    @Test
    void logTicker_logsSuccessfullyWithValidData() {
        Ticker ticker = buildTicker(CurrencyPair.BTC_USD, "85000.50", "85000.25", "85000.75", "1234.56");
        assertDoesNotThrow(() -> service.logTicker(CurrencyPair.BTC_USD, ticker));
    }

    // ---- Helper ----

    private Ticker buildTicker(CurrencyPair pair) {
        return buildTicker(pair, "100.00", "99.99", "100.01", "5000.00");
    }

    private Ticker buildTicker(CurrencyPair pair, String last, String bid, String ask, String volume) {
        return new Ticker.Builder()
                .currencyPair(pair)
                .last(last != null ? new BigDecimal(last) : null)
                .bid(bid != null ? new BigDecimal(bid) : null)
                .ask(ask != null ? new BigDecimal(ask) : null)
                .volume(volume != null ? new BigDecimal(volume) : null)
                .timestamp(new Date())
                .build();
    }
}
