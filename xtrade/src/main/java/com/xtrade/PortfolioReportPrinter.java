package com.xtrade;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.Instant;
import java.util.Map;

/**
 * Prints a formatted summary report of the portfolio to the console and logs.
 * <p>
 * Outputs a table with columns: Pair, Qty, Avg Cost, Current Price, Unrealized P&L, Total P&L.
 * Includes portfolio totals, return percentage, and trade count.
 */
public class PortfolioReportPrinter {

    private static final Logger LOG = LoggerFactory.getLogger(PortfolioReportPrinter.class);

    /** Column widths for the formatted table. */
    private static final int W_PAIR = 10;
    private static final int W_QTY = 14;
    private static final int W_AVG_COST = 14;
    private static final int W_CUR_PRICE = 14;
    private static final int W_UNRL_PNL = 16;
    private static final int W_TOTAL_PNL = 16;

    /**
     * Prints a complete portfolio report to both console (stdout) and the structured log file.
     *
     * @param snapshot the portfolio snapshot to report
     */
    public void printReport(PortfolioSnapshot snapshot) {
        String report = buildReport(snapshot);
        System.out.println(report);
        LOG.info("Portfolio report generated:\n{}", report);
    }

    /**
     * Builds the full report as a String. Useful for testing.
     *
     * @param snapshot the portfolio snapshot to report
     * @return formatted report string
     */
    public String buildReport(PortfolioSnapshot snapshot) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        printHeader(pw);
        printPositionRows(pw, snapshot);
        printSeparator(pw);
        printTotals(pw, snapshot);
        printFooter(pw, snapshot);

        pw.flush();
        return sw.toString();
    }

    // ------------------------------------------------------------------ //
    //  Header
    // ------------------------------------------------------------------ //

    private void printHeader(PrintWriter pw) {
        pw.println();
        pw.println("=====================================================================");
        pw.println("            PORTFOLIO SUMMARY REPORT");
        pw.printf("            Generated: %s%n", Instant.now());
        pw.println("=====================================================================");
        pw.println();

        // Column headers
        String headerLine = String.format(
                "  %-" + W_PAIR + "s %" + W_QTY + "s %" + W_AVG_COST + "s %" + W_CUR_PRICE + "s %"
                        + W_UNRL_PNL + "s %" + W_TOTAL_PNL + "s",
                "Pair", "Qty", "Avg Cost", "Cur Price", "Unrealized P&L", "Total P&L"
        );
        pw.println(headerLine);

        printSeparator(pw);
    }

    // ------------------------------------------------------------------ //
    //  Position rows
    // ------------------------------------------------------------------ //

    private void printPositionRows(PrintWriter pw, PortfolioSnapshot snapshot) {
        Map<String, PortfolioPosition> positions = snapshot.getPositions();

        for (Map.Entry<String, PortfolioPosition> entry : positions.entrySet()) {
            PortfolioPosition pos = entry.getValue();

            String pair = pos.getPair();
            double qty = pos.getQuantity();
            double avgCost = pos.getAverageCost();
            double curPrice = pos.getCurrentPrice();
            double unrealizedPnl = pos.getUnrealizedPnl();
            double realizedPnl = pos.getTotalRealizedPnl();
            double totalPnl = unrealizedPnl + realizedPnl - pos.getTotalFeesPaid();

            String qtyStr = qty > 0 ? formatQty(qty) : "-";
            String avgCostStr = qty > 0 ? formatMoney(avgCost) : "-";
            String curPriceStr = curPrice > 0 ? formatMoney(curPrice) : "-";
            String unrlPnlStr = qty > 0 ? formatPnl(unrealizedPnl) : "-";
            String totalPnlStr = qty > 0 || realizedPnl != 0 ? formatPnl(totalPnl) : "-";

            String row = String.format(
                    "  %-" + W_PAIR + "s %" + W_QTY + "s %" + W_AVG_COST + "s %" + W_CUR_PRICE + "s %"
                            + W_UNRL_PNL + "s %" + W_TOTAL_PNL + "s",
                    pair, qtyStr, avgCostStr, curPriceStr, unrlPnlStr, totalPnlStr
            );
            pw.println(row);
        }
    }

    // ------------------------------------------------------------------ //
    //  Totals
    // ------------------------------------------------------------------ //

    private void printTotals(PrintWriter pw, PortfolioSnapshot snapshot) {
        pw.println();

        double totalReturn = snapshot.getTotalPortfolioValueUsd()
                - snapshot.getCashBalance()
                + snapshot.getCashBalance();

        // Compute the starting balance from the snapshot context:
        // totalPortfolioValueUsd = cash + holdings value
        // totalReturn = totalUnrealizedPnl + totalRealizedPnl - totalFeesPaid
        double totalPnl = snapshot.getTotalUnrealizedPnl() + snapshot.getTotalRealizedPnl()
                - snapshot.getTotalFeesPaid();

        pw.printf("  Cash Balance:       %s%n", formatMoney(snapshot.getCashBalance()));
        pw.printf("  Holdings Value:     %s%n",
                formatMoney(snapshot.getTotalPortfolioValueUsd() - snapshot.getCashBalance()));
        pw.printf("  Total Portfolio:    %s%n", formatMoney(snapshot.getTotalPortfolioValueUsd()));
        pw.println();
        pw.printf("  Unrealized P&L:     %s%n", formatPnl(snapshot.getTotalUnrealizedPnl()));
        pw.printf("  Realized P&L:       %s%n", formatPnl(snapshot.getTotalRealizedPnl()));
        pw.printf("  Total Fees Paid:   %s%n", formatMoney(snapshot.getTotalFeesPaid()));
        pw.printf("  Net Total P&L:      %s%n", formatPnl(totalPnl));
        pw.println();
        pw.printf("  Total Trades:       %d%n", snapshot.getTradeHistory().size());
    }

    private void printFooter(PrintWriter pw, PortfolioSnapshot snapshot) {
        pw.println("=====================================================================");
        pw.println();
    }

    private void printSeparator(PrintWriter pw) {
        pw.printf("  %s%n",
                "-----------------------------------------------------------------------------"
                        + "----");
    }

    // ------------------------------------------------------------------ //
    //  Formatting helpers
    // ------------------------------------------------------------------ //

    private String formatMoney(double value) {
        if (Math.abs(value) >= 1000) {
            return String.format("$%,.2f", value);
        } else if (Math.abs(value) >= 1) {
            return String.format("$%.4f", value);
        } else {
            return String.format("$%.6f", value);
        }
    }

    private String formatQty(double qty) {
        if (qty >= 1000) {
            return String.format("%.2f", qty);
        } else if (qty >= 1) {
            return String.format("%.4f", qty);
        } else {
            return String.format("%.8f", qty);
        }
    }

    private String formatPnl(double pnl) {
        if (pnl >= 0) {
            return String.format("+$%,.4f", pnl);
        } else {
            return String.format("-$%,.4f", Math.abs(pnl));
        }
    }
}
