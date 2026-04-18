package com.xtrade.kline.binance;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public final class JdkBinanceHttpClient implements BinanceHttpClient {
    private final HttpClient client;

    public JdkBinanceHttpClient() {
        this(HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build());
    }

    JdkBinanceHttpClient(HttpClient client) {
        this.client = client;
    }

    @Override
    public String getString(URI uri) throws IOException, InterruptedException {
        HttpResponse<String> response = client.send(HttpRequest.newBuilder(uri).GET().build(), HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() == 404) {
            throw new IOException("404 Not Found: " + uri);
        }
        if (response.statusCode() >= 400) {
            throw new IOException("HTTP " + response.statusCode() + " for " + uri + ": " + response.body());
        }
        return response.body();
    }

    @Override
    public byte[] getBytes(URI uri) throws IOException, InterruptedException {
        HttpResponse<byte[]> response = client.send(HttpRequest.newBuilder(uri).GET().build(), HttpResponse.BodyHandlers.ofByteArray());
        if (response.statusCode() == 404) {
            throw new IOException("404 Not Found: " + uri);
        }
        if (response.statusCode() >= 400) {
            throw new IOException("HTTP " + response.statusCode() + " for " + uri);
        }
        return response.body();
    }
}
