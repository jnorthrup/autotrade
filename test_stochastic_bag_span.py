import json
import random
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb

import showdown
from showdown import (
    _choose_growth_dim,
    _initial_stagnation_metrics,
    _product_ids_with_window_data,
    _set_model_learning_rate,
    _stochastic_bag_limit,
    _stochastic_bag_sample,
    _stagnation_message,
    _stochastic_span_bars,
    _span_months_to_bars,
    format_bash_expansion,
)
from tools import fiat_connectome
from tools import import_binance_vision_pairs as ibv


class HighRNG:
    def randint(self, low, high):
        return high

    def choice(self, seq):
        return seq[0]

    def shuffle(self, seq):
        seq.reverse()


class StochasticBagSpanTests(unittest.TestCase):
    def test_choose_session_window_retries_until_one_valid_span_exists(self):
        pair_universe = [f"A{i}-USDT" for i in range(5)]

        with patch(
            "showdown._choose_stochastic_window",
            side_effect=[
                (48, datetime(2023, 7, 1), datetime(2023, 7, 2)),
                (96, datetime(2023, 7, 3), datetime(2023, 7, 5)),
            ],
        ), patch(
            "showdown._product_ids_with_window_data",
            side_effect=[
                ["A0-USDT", "A1-USDT"],
                pair_universe,
            ],
        ):
            session_window = showdown._choose_session_window(
                db_path="unused.duckdb",
                exchange="binance",
                pair_universe=pair_universe,
                model_size=1,
                rng=random.Random(0),
                bag_limit=50,
                db_min_ts=datetime(2023, 1, 1),
                total_seconds=1000.0,
                total_bars=1000,
            )

        self.assertIsNotNone(session_window)
        self.assertEqual(session_window.window_bars, 96)
        self.assertEqual(session_window.start_time, datetime(2023, 7, 3))
        self.assertEqual(session_window.end_time, datetime(2023, 7, 5))

    def test_build_trial_graph_uses_frozen_session_span_without_repick(self):
        pair_universe = [f"A{i}-USDT" for i in range(5)]

        class FakeCoinGraph:
            def __init__(self, fee_rate=0.001):
                self.fee_rate = fee_rate
                self.edges = {("coinbase", "A0", "USDT"): object()}
                self.common_timestamps = [datetime(2023, 7, 1)]
                self.all_pairs = []

            def load(
                self,
                *,
                db_path,
                granularity,
                exchange,
                skip_fetch,
                start_time,
                end_time,
                explicit_bag,
            ):
                self.all_pairs = [
                    f"{exchange}:{item['product_id']}" for item in explicit_bag
                ]
                self.edges = {
                    (exchange, item["product_id"].split("-", 1)[0], item["product_id"].split("-", 1)[1]): object()
                    for item in explicit_bag
                }
                self.common_timestamps = [start_time, end_time]

        with patch(
            "showdown._choose_stochastic_window",
            side_effect=AssertionError("session span should not be re-picked"),
        ), patch(
            "showdown._product_ids_with_window_data",
            return_value=pair_universe,
        ), patch(
            "showdown._stochastic_bag_sample",
            return_value=pair_universe,
        ), patch(
            "showdown.CoinGraph",
            FakeCoinGraph,
        ):
            built = showdown._build_stochastic_trial_graph(
                db_path="unused.duckdb",
                exchange="coinbase",
                pair_universe=pair_universe,
                model_size=1,
                rng=random.Random(0),
                bag_limit=50,
                db_min_ts=datetime(2023, 1, 1),
                total_seconds=1000.0,
                total_bars=1000,
                fixed_start_time=datetime(2023, 7, 1),
                fixed_end_time=datetime(2023, 7, 2),
                fixed_session_window_bars=48,
            )

        self.assertIsNotNone(built)
        self.assertEqual(built.window_bars, 48)
        self.assertEqual(built.selected_pairs, pair_universe)

    def test_dim_one_bag_stays_tiny(self):
        pairs = ["A-B", "B-C", "C-D", "D-E", "E-F", "F-G", "G-H"]
        selected = _stochastic_bag_sample(pairs, 1, random.Random(0), min_pairs=5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(pid in pairs for pid in selected))

    def test_width_scales_bag_target(self):
        pairs = [f"A{i}-B{i}" for i in range(40)]
        selected = _stochastic_bag_sample(pairs, 4, random.Random(0), min_pairs=5)
        self.assertEqual(len(selected), 20)

    def test_overlap_anchors_land_on_both_sides_when_target_allows(self):
        pairs = [
            "BTC-USDT",
            "ETH-USDT",
            "SOL-USDT",
            "ADA-BTC",
            "DOGE-BTC",
            "BNB-BTC",
            "ADA-ETH",
            "DOGE-ETH",
            "BNB-ETH",
            "ADA-SOL",
            "DOGE-SOL",
            "BNB-SOL",
            "BTC-ETH",
            "BTC-SOL",
            "ETH-BTC",
            "ETH-SOL",
            "SOL-BTC",
            "SOL-ETH",
            "XRP-USDT",
            "AVAX-USDT",
            "LINK-USDT",
            "MATIC-USDT",
        ]
        selected = _stochastic_bag_sample(
            pairs,
            4,
            random.Random(0),
            min_pairs=1,
            max_pairs=12,
            target_pairs=12,
        )

        roles = {asset: {"base": 0, "quote": 0} for asset in ("BTC", "ETH", "SOL")}
        for pair in selected:
            base, quote = pair.split("-", 1)
            if base in roles:
                roles[base]["base"] += 1
            if quote in roles:
                roles[quote]["quote"] += 1

        self.assertEqual(len(selected), 12)
        for asset in ("BTC", "ETH", "SOL"):
            self.assertGreaterEqual(roles[asset]["base"], 1)
            self.assertGreaterEqual(roles[asset]["quote"], 1)

    def test_default_span_bars_is_50(self):
        span = _stochastic_span_bars(1000, 4, HighRNG(), span_bars=50)
        self.assertEqual(span, 50)

    def test_span_grows_commensurately(self):
        self.assertEqual(_stochastic_span_bars(1000, 16, HighRNG(), span_bars=50), 100)
        self.assertEqual(_stochastic_span_bars(1000, 64, HighRNG(), span_bars=50), 200)

    def test_span_override_works(self):
        self.assertEqual(_stochastic_span_bars(1000, 4, HighRNG(), span_bars=25), 25)

    def test_span_months_converts_to_six_month_window(self):
        self.assertEqual(_span_months_to_bars(6), 52596)

    def test_non_autoresearch_bag_cap_is_50(self):
        self.assertEqual(_stochastic_bag_limit(64), 50)

    def test_non_autoresearch_bag_cap_override(self):
        self.assertEqual(_stochastic_bag_limit(64, cap=12), 12)

    def test_window_pair_lookup_filters_to_pairs_with_local_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "candles.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE candles (
                        exchange VARCHAR,
                        product_id VARCHAR,
                        timestamp TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO candles VALUES
                        ('binance', 'BTC-USDT', TIMESTAMP '2020-05-02 00:00:00'),
                        ('binance', 'ETH-USDT', TIMESTAMP '2020-05-03 00:00:00'),
                        ('binance', 'DOGE-USDT', TIMESTAMP '2021-05-03 00:00:00')
                    """
                )

            product_ids = _product_ids_with_window_data(
                str(db_path),
                "binance",
                datetime(2020, 5, 1),
                datetime(2020, 11, 1),
            )

            self.assertEqual(product_ids, ["BTC-USDT", "ETH-USDT"])

    def test_bag_contents_are_delineated(self):
        self.assertEqual(
            format_bash_expansion(["BTC-USDT", "ETH-USDT", "ADA-BTC"]),
            "[{BTC,ETH}-USDT | ADA-BTC]",
        )

    def test_bag_contents_compact_on_largest_side_of_component(self):
        self.assertEqual(
            format_bash_expansion(["ADA-BTC", "BTC-USDT", "ETH-USDT", "DOGE-BTC"]),
            "[{ADA,DOGE}-BTC | {BTC,ETH}-USDT]",
        )

    def test_bag_contents_compact_cartesian_group(self):
        self.assertEqual(
            format_bash_expansion(["BTC-USD", "BTC-USDT", "ETH-USD", "ETH-USDT"]),
            "[{BTC,ETH}-{USD,USDT}]",
        )

    def test_bag_contents_extract_largest_cartesian_before_fallback(self):
        self.assertEqual(
            format_bash_expansion(
                ["ADA-BTC", "ADA-USDT", "DOGE-BTC", "DOGE-USDT", "SOL-USDT"]
            ),
            "[{ADA,DOGE}-{BTC,USDT} | SOL-USDT]",
        )

    def test_growth_dim_pushes_all_ones_to_fours_before_sixteens(self):
        self.assertEqual(
            _choose_growth_dim(
                (("h", 4), ("H", 1), ("L", 4), ("Hc", 1), ("Lc", 4))
            ),
            "H",
        )
        self.assertEqual(
            _choose_growth_dim(
                (("h", 4), ("H", 4), ("L", 4), ("Hc", 4), ("Lc", 16))
            ),
            "h",
        )

    def test_flat_initial_losses_trigger_stagnation(self):
        stagnant, reason, improvement, observed = _initial_stagnation_metrics(
            [1.0] * 32, 32
        )
        self.assertTrue(stagnant)
        self.assertEqual(reason, "flat_loss")
        self.assertEqual(improvement, 0.0)
        self.assertEqual(observed, 32)

    def test_falling_initial_losses_do_not_trigger_stagnation(self):
        losses = [1.0 - (0.02 * idx) for idx in range(32)]
        stagnant, reason, improvement, observed = _initial_stagnation_metrics(
            losses, 32
        )
        self.assertFalse(stagnant)
        self.assertEqual(reason, "improving")
        self.assertGreater(improvement, 0.02)
        self.assertEqual(observed, 32)

    def test_trained_model_stagnation_warns_and_new_model_is_expected(self):
        trained_message = _stagnation_message(
            phase=3,
            old_lr=1e-4,
            new_lr=5e-4,
            reason="flat_loss",
            relative_improvement=0.0,
            observed_updates=32,
            resumed_from_checkpoint=True,
        )
        new_message = _stagnation_message(
            phase=1,
            old_lr=1e-4,
            new_lr=5e-4,
            reason="flat_loss",
            relative_improvement=0.0,
            observed_updates=32,
            resumed_from_checkpoint=False,
        )
        self.assertIn("WARNING", trained_message)
        self.assertIn("trained checkpoint", trained_message)
        self.assertIn("expected on a new model cold start", new_message)
        self.assertIn("INFO", new_message)

    def test_learning_rate_boost_updates_optimizer_groups(self):
        class DummyModel:
            def __init__(self):
                self._lr = 1e-4
                self._optimizer = type(
                    "DummyOptimizer",
                    (),
                    {"param_groups": [{"lr": self._lr}, {"lr": self._lr}]},
                )()

        model = DummyModel()
        self.assertEqual(_set_model_learning_rate(model, model._lr * 5.0), 5e-4)
        self.assertEqual(model._lr, 5e-4)
        self.assertEqual(
            [group["lr"] for group in model._optimizer.param_groups],
            [5e-4, 5e-4],
        )

    def test_exchangeinfo_pair_fetch_returns_real_pairs(self):
        payload = {
            "symbols": [
                {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "isSpotTradingAllowed": True,
                },
                {
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "isSpotTradingAllowed": True,
                },
                {
                    "baseAsset": "BTCUP",
                    "quoteAsset": "USDT",
                    "symbol": "BTCUPUSDT",
                    "status": "TRADING",
                    "isSpotTradingAllowed": True,
                },
                {
                    "baseAsset": "JUP",
                    "quoteAsset": "USDT",
                    "symbol": "JUPUSDT",
                    "status": "TRADING",
                    "isSpotTradingAllowed": True,
                },
            ]
        }

        class DummyResponse:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch(
            "urllib.request.urlopen",
            return_value=DummyResponse(json.dumps(payload).encode("utf-8")),
        ):
            self.assertEqual(
                ibv.collect_exchangeinfo_pairs(),
                [("BTC", "USDT"), ("ETH", "USDT"), ("JUP", "USDT")],
            )

    def test_symbol_split_uses_api_quotes_before_length_heuristics(self):
        context = ibv.build_symbol_split_context(
            [
                {"s": "BTCFDUSD", "b": "BTC", "q": "FDUSD", "pm": "ALTS", "pn": "ALTS"},
                {"s": "AAVEBTC", "b": "AAVE", "q": "BTC", "pm": "BTC", "pn": "BTC"},
                {"s": "JUPUSDT", "b": "JUP", "q": "USDT", "pm": "USDT", "pn": "USDT"},
                {
                    "s": "1INCHUSDT",
                    "b": "1INCH",
                    "q": "USDT",
                    "pm": "USDT",
                    "pn": "USDT",
                },
            ]
        )

        self.assertEqual(
            ibv.split_symbol("BTCFDUSD", context=context), ("BTC", "FDUSD")
        )
        self.assertEqual(ibv.split_symbol("AAVEBTC", context=context), ("AAVE", "BTC"))
        self.assertEqual(
            ibv.split_symbol("1INCHDOWNUSDT", context=context),
            ("1INCHDOWN", "USDT"),
        )

    def test_pair_record_builder_marks_leveraged_residue_keep_false(self):
        context = ibv.build_symbol_split_context(
            [
                {"s": "BTCUSDT", "b": "BTC", "q": "USDT", "pm": "USDT", "pn": "USDT"},
                {"s": "JUPUSDT", "b": "JUP", "q": "USDT", "pm": "USDT", "pn": "USDT"},
                {
                    "s": "1INCHUSDT",
                    "b": "1INCH",
                    "q": "USDT",
                    "pm": "USDT",
                    "pn": "USDT",
                },
                {"s": "FETFDUSD", "b": "FET", "q": "FDUSD", "pm": "ALTS", "pn": "ALTS"},
            ]
        )

        records = ibv.build_pair_records_from_symbols(
            ["JUPUSDT", "1INCHDOWNUSDT", "BTCUSDT", "FETFDUSD"],
            context=context,
        )
        by_symbol = {record["symbol"]: record for record in records}

        self.assertTrue(by_symbol["JUPUSDT"]["keep"])
        self.assertFalse(by_symbol["JUPUSDT"]["leveraged"])
        self.assertTrue(by_symbol["FETFDUSD"]["keep"])
        self.assertFalse(by_symbol["FETFDUSD"]["is_etf"])
        self.assertFalse(by_symbol["1INCHDOWNUSDT"]["keep"])
        self.assertTrue(by_symbol["1INCHDOWNUSDT"]["leveraged"])
        self.assertEqual(by_symbol["1INCHDOWNUSDT"]["base"], "1INCHDOWN")
        self.assertEqual(by_symbol["1INCHDOWNUSDT"]["quote"], "USDT")

    def test_coin_record_builder_aggregates_roles_and_quote_types(self):
        records = ibv.build_coin_records_from_pair_records(
            "binance",
            [
                {"base": "DOGE", "quote": "USD", "quote_type": "fiat", "keep": True},
                {
                    "base": "DOGE",
                    "quote": "USDT",
                    "quote_type": "stablecoin",
                    "keep": True,
                },
                {"base": "DOGE", "quote": "BTC", "quote_type": "crypto", "keep": True},
                {"base": "AAVE", "quote": "BTC", "quote_type": "crypto", "keep": True},
                {
                    "base": "BTCUP",
                    "quote": "USDT",
                    "quote_type": "stablecoin",
                    "keep": False,
                },
            ],
        )
        by_asset = {record["asset"]: record for record in records}

        self.assertEqual(by_asset["DOGE"]["base_pair_count"], 3)
        self.assertEqual(by_asset["DOGE"]["fiat_base_count"], 1)
        self.assertEqual(by_asset["DOGE"]["stable_base_count"], 1)
        self.assertEqual(by_asset["DOGE"]["crypto_base_count"], 1)
        self.assertTrue(by_asset["DOGE"]["has_fiat_base"])
        self.assertTrue(by_asset["DOGE"]["has_stable_base"])
        self.assertTrue(by_asset["DOGE"]["has_crypto_base"])
        self.assertEqual(by_asset["BTC"]["quote_pair_count"], 2)
        self.assertTrue(by_asset["BTC"]["is_counterquote"])
        self.assertNotIn("BTCUP", by_asset)

    def test_list_pairs_table_subscriptions_only_returns_keep_true_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "pairs.duckdb")
            conn = duckdb.connect(db_path)
            ibv.ensure_pairs_schema(conn)
            conn.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC-USDT",
                        "BTC",
                        "USDT",
                        "stablecoin",
                        False,
                        False,
                        True,
                    ],
                    [
                        "binance",
                        "1INCHDOWN-USDT",
                        "1INCHDOWN",
                        "USDT",
                        "stablecoin",
                        True,
                        False,
                        False,
                    ],
                ],
            )
            conn.close()

            with patch("showdown._use_pool_for_db", return_value=False):
                subscriptions = showdown._list_pairs_table_subscriptions(
                    db_path, exchange="binance"
                )

            self.assertEqual(
                subscriptions,
                [{"exchange": "binance", "product_id": "BTC-USDT"}],
            )

    def test_list_pairs_table_subscriptions_filters_shitlisted_coins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "pairs.duckdb")
            conn = duckdb.connect(db_path)
            ibv.ensure_pairs_schema(conn)
            ibv.ensure_coins_schema(conn)
            conn.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC-USDT",
                        "BTC",
                        "USDT",
                        "stablecoin",
                        False,
                        False,
                        True,
                    ],
                    [
                        "binance",
                        "AAVE-BTC",
                        "AAVE",
                        "BTC",
                        "crypto",
                        False,
                        False,
                        True,
                    ],
                ],
            )
            conn.executemany(
                "INSERT INTO coins VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC",
                        2,
                        1,
                        1,
                        0,
                        1,
                        0,
                        False,
                        True,
                        False,
                        True,
                        True,
                        "too central",
                    ],
                    [
                        "binance",
                        "USDT",
                        1,
                        0,
                        1,
                        0,
                        0,
                        0,
                        False,
                        False,
                        False,
                        True,
                        False,
                        "",
                    ],
                    [
                        "binance",
                        "AAVE",
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        False,
                        False,
                        True,
                        False,
                        False,
                        "",
                    ],
                ],
            )
            conn.close()

            with patch("showdown._use_pool_for_db", return_value=False):
                subscriptions = showdown._list_pairs_table_subscriptions(
                    db_path, exchange="binance"
                )

            self.assertEqual(subscriptions, [])

    def test_mark_pairs_keep_removes_pair_from_runtime_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "pairs.duckdb")
            conn = duckdb.connect(db_path)
            ibv.ensure_pairs_schema(conn)
            ibv.ensure_coins_schema(conn)
            conn.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC-USDT",
                        "BTC",
                        "USDT",
                        "stablecoin",
                        False,
                        False,
                        True,
                    ],
                    [
                        "binance",
                        "AAVE-BTC",
                        "AAVE",
                        "BTC",
                        "crypto",
                        False,
                        False,
                        True,
                    ],
                ],
            )
            conn.close()

            ibv.refresh_coin_records_from_db(db_path, "binance")
            ibv.mark_pairs_keep(db_path, "binance", ["AAVE-BTC"], keep=False)

            with patch("showdown._use_pool_for_db", return_value=False):
                subscriptions = showdown._list_pairs_table_subscriptions(
                    db_path, exchange="binance"
                )

            self.assertEqual(
                subscriptions,
                [{"exchange": "binance", "product_id": "BTC-USDT"}],
            )

    def test_prefetch_window_miss_does_not_mutate_pair_keep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "pairs.duckdb")
            conn = duckdb.connect(db_path)
            ibv.ensure_pairs_schema(conn)
            conn.execute(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ["binance", "DOGE-U", "DOGE", "U", "crypto", False, False, True],
            )
            conn.close()

            with patch(
                "binance_cache.fetch_binance_vision",
                return_value={"pair": "DOGE-U", "got_any": False},
            ):
                invalid = showdown._prefetch_binance_candles_window(
                    db_path,
                    ["DOGE-U"],
                    datetime(2018, 12, 1, 0, 0),
                    datetime(2018, 12, 11, 0, 0),
                    granularity="300",
                )

            self.assertEqual(invalid, ["DOGE-U"])
            with duckdb.connect(db_path, read_only=True) as conn:
                keep = conn.execute(
                    "SELECT keep FROM pairs WHERE exchange = 'binance' AND product_id = 'DOGE-U'"
                ).fetchone()[0]
            self.assertTrue(keep)

    def test_connectome_loader_respects_keep_and_shitlist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pairs.duckdb"
            conn = duckdb.connect(str(db_path))
            ibv.ensure_pairs_schema(conn)
            ibv.ensure_coins_schema(conn)
            conn.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC-USDT",
                        "BTC",
                        "USDT",
                        "stablecoin",
                        False,
                        False,
                        True,
                    ],
                    [
                        "binance",
                        "AAVE-BTC",
                        "AAVE",
                        "BTC",
                        "crypto",
                        False,
                        False,
                        False,
                    ],
                    [
                        "binance",
                        "DOGE-USDT",
                        "DOGE",
                        "USDT",
                        "stablecoin",
                        False,
                        False,
                        True,
                    ],
                ],
            )
            conn.executemany(
                "INSERT INTO coins VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        "binance",
                        "BTC",
                        1,
                        1,
                        0,
                        0,
                        1,
                        0,
                        False,
                        True,
                        False,
                        True,
                        False,
                        "",
                    ],
                    [
                        "binance",
                        "USDT",
                        2,
                        0,
                        2,
                        0,
                        0,
                        0,
                        False,
                        False,
                        False,
                        True,
                        False,
                        "",
                    ],
                    [
                        "binance",
                        "AAVE",
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        False,
                        False,
                        True,
                        False,
                        False,
                        "",
                    ],
                    [
                        "binance",
                        "DOGE",
                        1,
                        1,
                        0,
                        0,
                        1,
                        0,
                        False,
                        True,
                        False,
                        False,
                        True,
                        "junk",
                    ],
                ],
            )
            conn.close()

            loaded = fiat_connectome.load_pairs_from_db(db_path, exchange="binance")

            self.assertEqual(loaded, [("BTC", "USDT")])

    def test_refill_binance_bag_replaces_window_misses_without_shrinking(self):
        initial = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-BTC"]

        def fake_prefetch(*args, **kwargs):
            selected_pairs = args[1]
            return ["ADA-BTC"] if "ADA-BTC" in selected_pairs else []

        with (
            patch(
                "showdown._prefetch_binance_candles_window",
                side_effect=fake_prefetch,
            ),
            patch(
                "showdown._stochastic_bag_sample",
                return_value=["DOGE-BTC"],
            ),
        ):
            refilled = showdown._refill_binance_stochastic_bag(
                "candles.duckdb",
                [
                    "BTC-USDT",
                    "ETH-USDT",
                    "SOL-USDT",
                    "ADA-BTC",
                    "DOGE-BTC",
                    "BNB-ETH",
                ],
                initial,
                bag_target=4,
                model_size=4,
                rng=random.Random(0),
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2),
            )

        self.assertEqual(
            refilled,
            ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-BTC"],
        )


if __name__ == "__main__":
    unittest.main()
