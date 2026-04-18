package com.xtrade.kline;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Objects;

/**
 * Lightweight HTTP monitor that exposes feed health and metrics endpoints.
 */
public final class KlineFeedMonitorServer implements AutoCloseable {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private final HttpServer server;

    private KlineFeedMonitorServer(HttpServer server) {
        this.server = server;
    }

    public static KlineFeedMonitorServer start(DrawThruCachingKlineFeed feed,
                                               InetSocketAddress address,
                                               Duration staleAfter) throws IOException {
        Objects.requireNonNull(feed, "feed must not be null");
        Objects.requireNonNull(address, "address must not be null");
        Objects.requireNonNull(staleAfter, "staleAfter must not be null");
        HttpServer server = HttpServer.create(address, 0);
        server.createContext("/health", exchange -> writeJson(exchange, feed.healthReport(staleAfter)));
        server.createContext("/metrics", exchange -> writeText(exchange, feed.prometheusMetrics(staleAfter), "text/plain; version=0.0.4"));
        server.setExecutor(null);
        server.start();
        return new KlineFeedMonitorServer(server);
    }

    public int port() {
        return server.getAddress().getPort();
    }

    @Override
    public void close() {
        server.stop(0);
    }

    private static void writeJson(HttpExchange exchange, Object payload) throws IOException {
        writeText(exchange, GSON.toJson(payload), "application/json");
    }

    private static void writeText(HttpExchange exchange, String payload, String contentType) throws IOException {
        byte[] body = payload.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", contentType + "; charset=utf-8");
        exchange.sendResponseHeaders(200, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }
}
