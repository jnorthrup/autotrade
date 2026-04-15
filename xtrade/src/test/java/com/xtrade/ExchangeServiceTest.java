package com.xtrade;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.dto.Order;
import org.knowm.xchange.dto.marketdata.OrderBook;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.dto.trade.LimitOrder;
import org.knowm.xchange.service.marketdata.MarketDataService;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ExchangeServiceTest {

    @Mock
    private Exchange exchange;

    @Mock
    private MarketDataService marketDataService;

    private ExchangeService service;

    @BeforeEach
    void setUp() {
        when(exchange.getMarketDataService()).thenReturn(marketDataService);
        // Use 0 rate-limit gap and short retry delay for fast tests
        service = new ExchangeService(exchange, 3, 10L, 0L);
    }

    // ------------------------------------------------------------------ //
    //                     Constructor validation                          //
    // ------------------------------------------------------------------ //

    @Test
    void constructor_rejectsNullExchange() {
        assertThrows(NullPointerException.class,
                () -> new ExchangeService((Exchange) null, 3, 10L, 0L));
    }

    @Test
    void constructor_rejectsNegativeMaxRetries() {
        AppConfig cfg = new AppConfig("key", "dGVzdHNlY3JldA==", null, new Properties());
        assertThrows(IllegalArgumentException.class,
                () -> new ExchangeService(cfg, -1, 10L, 0L));
    }

    @Test
    void constructor_rejectsZeroRetryDelay() {
        AppConfig cfg = new AppConfig("key", "dGVzdHNlY3JldA==", null, new Properties());
        assertThrows(IllegalArgumentException.class,
                () -> new ExchangeService(cfg, 3, 0L, 0L));
    }

    @Test
    void constructor_fromAppConfig_createsKrakenExchange() {
        // Kraken secret must be valid base64
        AppConfig cfg = new AppConfig("test-key-abc", "dGVzdHNlY3JldGtleQ==", null, new Properties());
        ExchangeService svc = new ExchangeService(cfg);
        assertNotNull(svc.getExchange());
        assertInstanceOf(org.knowm.xchange.kraken.KrakenExchange.class, svc.getExchange());
    }

    @Test
    void constructor_accessorsReturnConfiguredValues() {
        assertEquals(3, service.getMaxRetries());
        assertEquals(0L, service.getRateLimitGapMs());
    }

    // ------------------------------------------------------------------ //
    //                        getCurrentTicker                             //
    // ------------------------------------------------------------------ //

    @Test
    void getCurrentTicker_returnsValidTicker() throws Exception {
        Ticker expected = buildTicker(CurrencyPair.BTC_USD, "85000.50", "85000.25", "85000.75", "1234.56");
        when(marketDataService.getTicker(CurrencyPair.BTC_USD)).thenReturn(expected);

        Ticker result = service.getCurrentTicker(CurrencyPair.BTC_USD);

        assertNotNull(result);
        assertEquals(new BigDecimal("85000.50"), result.getLast());
        assertEquals(new BigDecimal("85000.25"), result.getBid());
        assertEquals(new BigDecimal("85000.75"), result.getAsk());
        assertEquals(new BigDecimal("1234.56"), result.getVolume());
    }

    @Test
    void getCurrentTicker_rejectsNullPair() {
        assertThrows(NullPointerException.class, () -> service.getCurrentTicker(null));
    }

    @Test
    void getCurrentTicker_retriesOnTransientError() throws Exception {
        Ticker ticker = buildTicker(CurrencyPair.BTC_USD);
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("Connection reset"))
                .thenReturn(ticker);

        Ticker result = service.getCurrentTicker(CurrencyPair.BTC_USD);

        assertNotNull(result);
        verify(marketDataService, times(2)).getTicker(CurrencyPair.BTC_USD);
    }

    @Test
    void getCurrentTicker_retriesMultipleTimes() throws Exception {
        Ticker ticker = buildTicker(CurrencyPair.BTC_USD);
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("timeout 1"))
                .thenThrow(new IOException("timeout 2"))
                .thenReturn(ticker);

        Ticker result = service.getCurrentTicker(CurrencyPair.BTC_USD);

        assertNotNull(result);
        verify(marketDataService, times(3)).getTicker(CurrencyPair.BTC_USD);
    }

    @Test
    void getCurrentTicker_throwsAfterMaxRetriesExhausted() throws Exception {
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("persistent failure"));

        IOException ex = assertThrows(IOException.class,
                () -> service.getCurrentTicker(CurrencyPair.BTC_USD));
        assertTrue(ex.getMessage().contains("persistent failure"));
        verify(marketDataService, times(4)).getTicker(CurrencyPair.BTC_USD); // 1 initial + 3 retries
    }

    @Test
    void getCurrentTicker_throwsOnNullTicker() throws Exception {
        when(marketDataService.getTicker(CurrencyPair.BTC_USD)).thenReturn(null);

        IOException ex = assertThrows(IOException.class,
                () -> service.getCurrentTicker(CurrencyPair.BTC_USD));
        assertTrue(ex.getMessage().contains("null ticker"));
    }

    // ------------------------------------------------------------------ //
    //                        getOrderBook                                 //
    // ------------------------------------------------------------------ //

    @Test
    void getOrderBook_returnsBidsAndAsks() throws Exception {
        OrderBook orderBook = buildOrderBook(CurrencyPair.BTC_USD, 5);
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 5)).thenReturn(orderBook);

        OrderBook result = service.getOrderBook(CurrencyPair.BTC_USD, 5);

        assertNotNull(result);
        assertEquals(5, result.getBids().size());
        assertEquals(5, result.getAsks().size());
    }

    @Test
    void getOrderBook_withDepthZero_usesNoArgsVersion() throws Exception {
        OrderBook orderBook = buildOrderBook(CurrencyPair.ETH_USD, 3);
        when(marketDataService.getOrderBook(CurrencyPair.ETH_USD)).thenReturn(orderBook);

        OrderBook result = service.getOrderBook(CurrencyPair.ETH_USD, 0);

        assertNotNull(result);
        assertEquals(3, result.getBids().size());
        verify(marketDataService).getOrderBook(CurrencyPair.ETH_USD);
        verify(marketDataService, never()).getOrderBook(any(CurrencyPair.class), anyInt());
    }

    @Test
    void getOrderBook_rejectsNullPair() {
        assertThrows(NullPointerException.class, () -> service.getOrderBook(null, 10));
    }

    @Test
    void getOrderBook_retriesOnTransientError() throws Exception {
        OrderBook ob = buildOrderBook(CurrencyPair.BTC_USD, 3);
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 10))
                .thenThrow(new IOException("network error"))
                .thenReturn(ob);

        OrderBook result = service.getOrderBook(CurrencyPair.BTC_USD, 10);

        assertNotNull(result);
        verify(marketDataService, times(2)).getOrderBook(CurrencyPair.BTC_USD, 10);
    }

    @Test
    void getOrderBook_throwsOnNullOrderBook() throws Exception {
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 10)).thenReturn(null);

        IOException ex = assertThrows(IOException.class,
                () -> service.getOrderBook(CurrencyPair.BTC_USD, 10));
        assertTrue(ex.getMessage().contains("null order book"));
    }

    @Test
    void getOrderBook_throwsAfterMaxRetries() throws Exception {
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 10))
                .thenThrow(new IOException("service unavailable"));

        assertThrows(IOException.class, () -> service.getOrderBook(CurrencyPair.BTC_USD, 10));
        verify(marketDataService, times(4)).getOrderBook(CurrencyPair.BTC_USD, 10);
    }

    @Test
    void getOrderBook_bidsContainCorrectData() throws Exception {
        OrderBook ob = buildOrderBook(CurrencyPair.BTC_USD, 2);
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 2)).thenReturn(ob);

        OrderBook result = service.getOrderBook(CurrencyPair.BTC_USD, 2);

        List<LimitOrder> bids = result.getBids();
        assertEquals(2, bids.size());
        // Bids are sorted descending by price in our helper
        assertTrue(bids.get(0).getLimitPrice().compareTo(bids.get(1).getLimitPrice()) >= 0);
        for (LimitOrder bid : bids) {
            assertNotNull(bid.getLimitPrice());
            assertNotNull(bid.getOriginalAmount());
            assertEquals(CurrencyPair.BTC_USD, bid.getCurrencyPair());
        }
    }

    // ------------------------------------------------------------------ //
    //                      getAllTickers                                  //
    // ------------------------------------------------------------------ //

    @Test
    void getAllTickers_returnsAllFivePairs() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        Map<TradingPair, Ticker> result = service.getAllTickers();

        assertEquals(5, result.size());
        for (TradingPair tp : TradingPair.values()) {
            assertNotNull(result.get(tp), "Missing ticker for " + tp);
        }
    }

    @Test
    void getAllTickers_continuesOnSinglePairFailure() throws Exception {
        // All pairs return a ticker except BTC_USD which always throws
        lenient().when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("API timeout"));

        Map<TradingPair, Ticker> result = service.getAllTickers();

        assertEquals(4, result.size());
        assertFalse(result.containsKey(TradingPair.BTC_USD));
        assertTrue(result.containsKey(TradingPair.ETH_USD));
    }

    @Test
    void getAllTickers_resultIsUnmodifiable() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        Map<TradingPair, Ticker> result = service.getAllTickers();

        assertThrows(UnsupportedOperationException.class,
                () -> result.put(TradingPair.BTC_USD, null));
    }

    // ------------------------------------------------------------------ //
    //                    getAllOrderBooks                                 //
    // ------------------------------------------------------------------ //

    @Test
    void getAllOrderBooks_returnsAllFivePairs() throws Exception {
        when(marketDataService.getOrderBook(any(CurrencyPair.class), eq(10)))
                .thenAnswer(invocation -> buildOrderBook(invocation.getArgument(0), 5));

        Map<TradingPair, OrderBook> result = service.getAllOrderBooks(10);

        assertEquals(5, result.size());
        for (TradingPair tp : TradingPair.values()) {
            OrderBook ob = result.get(tp);
            assertNotNull(ob, "Missing order book for " + tp);
            assertFalse(ob.getBids().isEmpty());
            assertFalse(ob.getAsks().isEmpty());
        }
    }

    @Test
    void getAllOrderBooks_continuesOnSinglePairFailure() throws Exception {
        // All pairs return an order book except BTC_USD which always throws
        lenient().when(marketDataService.getOrderBook(any(CurrencyPair.class), eq(5)))
                .thenAnswer(invocation -> buildOrderBook(invocation.getArgument(0), 3));
        when(marketDataService.getOrderBook(CurrencyPair.BTC_USD, 5))
                .thenThrow(new IOException("API error"));

        Map<TradingPair, OrderBook> result = service.getAllOrderBooks(5);

        assertEquals(4, result.size());
        assertFalse(result.containsKey(TradingPair.BTC_USD));
        assertTrue(result.containsKey(TradingPair.ETH_USD));
    }

    // ------------------------------------------------------------------ //
    //                      All 5 pairs individually                       //
    // ------------------------------------------------------------------ //

    @Test
    void getCurrentTicker_allFivePairs_succeed() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        for (TradingPair tp : TradingPair.values()) {
            Ticker ticker = service.getCurrentTicker(tp.getCurrencyPair());
            assertNotNull(ticker, "Null ticker for " + tp);
            assertNotNull(ticker.getLast(), "Null last price for " + tp);
            assertNotNull(ticker.getBid(), "Null bid for " + tp);
            assertNotNull(ticker.getAsk(), "Null ask for " + tp);
            assertNotNull(ticker.getVolume(), "Null volume for " + tp);
        }
    }

    @Test
    void getOrderBook_allFivePairs_succeed() throws Exception {
        when(marketDataService.getOrderBook(any(CurrencyPair.class), eq(5)))
                .thenAnswer(invocation -> buildOrderBook(invocation.getArgument(0), 5));

        for (TradingPair tp : TradingPair.values()) {
            OrderBook ob = service.getOrderBook(tp.getCurrencyPair(), 5);
            assertNotNull(ob, "Null order book for " + tp);
            assertFalse(ob.getBids().isEmpty(), "Empty bids for " + tp);
            assertFalse(ob.getAsks().isEmpty(), "Empty asks for " + tp);
        }
    }

    // ------------------------------------------------------------------ //
    //                    Rate-limit awareness                             //
    // ------------------------------------------------------------------ //

    @Test
    void enforceRateLimit_sleepsWhenGapNotElapsed() {
        ExchangeService svc = new ExchangeService(exchange, 0, 10L, 100L);
        // First call sets the timestamp; second call should trigger a sleep
        // (we can't easily test the sleep duration but we can verify no exception)
        assertDoesNotThrow(() -> svc.enforceRateLimit());
        assertDoesNotThrow(() -> svc.enforceRateLimit());
    }

    @Test
    void enforceRateLimit_noSleepWhenGapIsZero() {
        assertDoesNotThrow(() -> service.enforceRateLimit());
    }

    // ------------------------------------------------------------------ //
    //                       Retry logic edge cases                        //
    // ------------------------------------------------------------------ //

    @Test
    void executeWithRetry_succeedsOnFirstAttempt() throws Exception {
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenReturn(buildTicker(CurrencyPair.BTC_USD));

        Ticker result = service.getCurrentTicker(CurrencyPair.BTC_USD);

        verify(marketDataService, times(1)).getTicker(CurrencyPair.BTC_USD);
        assertNotNull(result);
    }

    @Test
    void executeWithRetry_zeroRetries_throwsImmediately() throws Exception {
        ExchangeService noRetrySvc = new ExchangeService(exchange, 0, 10L, 0L);
        when(marketDataService.getTicker(CurrencyPair.BTC_USD))
                .thenThrow(new IOException("fail"));

        assertThrows(IOException.class,
                () -> noRetrySvc.getCurrentTicker(CurrencyPair.BTC_USD));
        verify(marketDataService, times(1)).getTicker(CurrencyPair.BTC_USD);
    }

    // ------------------------------------------------------------------ //
    //                      Supported pairs                                //
    // ------------------------------------------------------------------ //

    @Test
    void getSupportedPairs_returnsAllFive() {
        List<TradingPair> pairs = service.getSupportedPairs();
        assertEquals(5, pairs.size());
        assertEquals(TradingPair.BTC_USD, pairs.get(0));
        assertEquals(TradingPair.ETH_USD, pairs.get(1));
        assertEquals(TradingPair.XRP_USD, pairs.get(2));
        assertEquals(TradingPair.SOL_USD, pairs.get(3));
        assertEquals(TradingPair.ADA_USD, pairs.get(4));
    }

    // ------------------------------------------------------------------ //
    //                       Log helper                                    //
    // ------------------------------------------------------------------ //

    @Test
    void logTicker_doesNotThrowWithNullFields() {
        Ticker ticker = buildTicker(CurrencyPair.BTC_USD, null, null, null, null);
        assertDoesNotThrow(() -> service.logTicker(CurrencyPair.BTC_USD, ticker));
    }

    @Test
    void logTicker_logsValidData() {
        Ticker ticker = buildTicker(CurrencyPair.ETH_USD, "3000.00", "2999.50", "3000.50", "50000");
        assertDoesNotThrow(() -> service.logTicker(CurrencyPair.ETH_USD, ticker));
    }

    // ------------------------------------------------------------------ //
    //                    validateAllPairs                                 //
    // ------------------------------------------------------------------ //

    @Test
    void validateAllPairs_returnsTrueWhenAllSucceed() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenAnswer(invocation -> buildTicker(invocation.getArgument(0)));

        assertTrue(service.validateAllPairs());
    }

    @Test
    void validateAllPairs_returnsFalseOnFailure() throws Exception {
        when(marketDataService.getTicker(any(CurrencyPair.class)))
                .thenThrow(new IOException("down"));

        assertFalse(service.validateAllPairs());
    }

    // ------------------------------------------------------------------ //
    //                       sleepQuietly                                  //
    // ------------------------------------------------------------------ //

    @Test
    void sleepQuietly_doesNotThrow() {
        assertDoesNotThrow(() -> ExchangeService.sleepQuietly(1));
    }

    // ------------------------------------------------------------------ //
    //                       Helpers                                       //
    // ------------------------------------------------------------------ //

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

    private OrderBook buildOrderBook(CurrencyPair pair, int depth) {
        List<LimitOrder> bids = new ArrayList<>();
        List<LimitOrder> asks = new ArrayList<>();

        BigDecimal basePrice = new BigDecimal("50000.00");
        for (int i = 0; i < depth; i++) {
            BigDecimal bidPrice = basePrice.subtract(new BigDecimal(i * 10));
            BigDecimal askPrice = basePrice.add(new BigDecimal(i * 10 + 5));
            BigDecimal amount = new BigDecimal("1.5").subtract(new BigDecimal(i * 0.1));

            bids.add(new LimitOrder(Order.OrderType.BID, amount, pair, null, null, bidPrice));
            asks.add(new LimitOrder(Order.OrderType.ASK, amount, pair, null, null, askPrice));
        }

        return new OrderBook(new Date(), asks, bids);
    }
}
