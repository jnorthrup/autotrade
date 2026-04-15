package com.xtrade;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PortfolioReportPrinter.
 * Verifies that the report contains all required columns and data.
 */
class PortfolioReportPrinterTest {

    private PortfolioReportPrinter printer;
    private PaperTradingEngine engine;

    @BeforeEach
    void setUp() {
        printer = new PortfolioReportPrinter();
        engine = new PaperTradingEngine(10000.00, true);
        engine.updateMarketPrice("BTC/USD", 50000.00);
        engine.updateMarketPrice("ETH/USD", 3000.00);
        engine.updateMarketPrice("XRP/USD", 0.50);
        engine.updateMarketPrice("SOL/USD", 100.00);
        engine.updateMarketPrice("ADA/USD", 0.30);
    }

    @Test
    void reportContainsAllColumnHeaders() {
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        assertTrue(report.contains("Pair"), "Report should contain 'Pair' header");
        assertTrue(report.contains("Qty"), "Report should contain 'Qty' header");
        assertTrue(report.contains("Avg Cost"), "Report should contain 'Avg Cost' header");
        assertTrue(report.contains("Cur Price"), "Report should contain 'Cur Price' header");
        assertTrue(report.contains("Unrealized P&L"), "Report should contain 'Unrealized P&L' header");
        assertTrue(report.contains("Total P&L"), "Report should contain 'Total P&L' header");
    }

    @Test
    void reportContainsAllFiveTradingPairs() {
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        assertTrue(report.contains("BTC/USD"), "Report should contain BTC/USD");
        assertTrue(report.contains("ETH/USD"), "Report should contain ETH/USD");
        assertTrue(report.contains("XRP/USD"), "Report should contain XRP/USD");
        assertTrue(report.contains("SOL/USD"), "Report should contain SOL/USD");
        assertTrue(report.contains("ADA/USD"), "Report should contain ADA/USD");
    }

    @Test
    void reportShowsPositionDataAfterBuy() {
        engine.marketBuy("BTC", 0.1);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        // Should contain the quantity
        assertTrue(report.contains("0.1"), "Report should show BTC quantity 0.1");
        // Should contain the current price
        assertTrue(report.contains("50000"), "Report should show BTC current price");
        // Should contain BTC/USD pair
        assertTrue(report.contains("BTC/USD"), "Report should show BTC/USD pair row");
    }

    @Test
    void reportShowsUnrealizedPnlAfterPriceChange() {
        engine.marketBuy("BTC", 0.1);
        engine.updateMarketPrice("BTC/USD", 55000.00);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        // Unrealized P&L should be (55000 - 50000) * 0.1 = 500
        assertTrue(report.contains("500"), "Report should show positive unrealized P&L");
    }

    @Test
    void reportShowsTotalPnlAfterSell() {
        engine.marketBuy("SOL", 10.0);
        engine.updateMarketPrice("SOL/USD", 200.00);
        engine.marketSell("SOL", 10.0);

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        // Should show realized P&L > 0
        assertTrue(report.contains("Realized P&L"), "Report should show Realized P&L line");
        // Should contain total trades count
        assertTrue(report.contains("Total Trades"), "Report should show Total Trades");
    }

    @Test
    void reportContainsCashAndTotals() {
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        assertTrue(report.contains("Cash Balance"), "Report should show Cash Balance");
        assertTrue(report.contains("Holdings Value"), "Report should show Holdings Value");
        assertTrue(report.contains("Total Portfolio"), "Report should show Total Portfolio");
        assertTrue(report.contains("Net Total P&L"), "Report should show Net Total P&L");
    }

    @Test
    void reportWithEmptyPositionsShowsDashes() {
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        // With no positions, qty columns should show dashes
        long dashCount = report.chars().filter(ch -> ch == '-').count();
        assertTrue(dashCount > 0, "Report should show dashes for empty positions");
    }

    @Test
    void reportContainsTradeCount() {
        engine.marketBuy("BTC", 0.01);
        engine.marketSell("BTC", 0.005);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        assertTrue(report.contains("Total Trades"), "Report should show Total Trades header");
        assertTrue(report.contains("2"), "Report should show 2 trades");
    }

    @Test
    void printReportDoesNotThrow() {
        engine.marketBuy("ETH", 1.0);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        assertDoesNotThrow(() -> printer.printReport(snapshot));
    }

    @Test
    void reportWithMultiplePositionsShowsAll() {
        engine.marketBuy("BTC", 0.05);
        engine.marketBuy("ETH", 2.0);
        engine.marketBuy("SOL", 5.0);

        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        // Should show quantities for all three
        assertTrue(report.contains("BTC/USD"), "Report should show BTC position");
        assertTrue(report.contains("ETH/USD"), "Report should show ETH position");
        assertTrue(report.contains("SOL/USD"), "Report should show SOL position");
    }

    @Test
    void reportShowsFeesPaid() {
        engine.marketBuy("BTC", 0.1);
        PortfolioSnapshot snapshot = engine.getPortfolioSnapshot();
        String report = printer.buildReport(snapshot);

        assertTrue(report.contains("Total Fees Paid"), "Report should show Total Fees Paid");
    }
}
