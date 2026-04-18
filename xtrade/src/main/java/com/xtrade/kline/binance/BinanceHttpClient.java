package com.xtrade.kline.binance;

import java.io.IOException;
import java.net.URI;

public interface BinanceHttpClient {
    String getString(URI uri) throws IOException, InterruptedException;

    byte[] getBytes(URI uri) throws IOException, InterruptedException;
}
