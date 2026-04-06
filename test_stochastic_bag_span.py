import random
import unittest

from showdown import _stochastic_bag_limit, _stochastic_bag_sample, _stochastic_span_bars


class HighRNG:
    def randint(self, low, high):
        return high

    def choice(self, seq):
        return seq[0]

    def shuffle(self, seq):
        seq.reverse()


class StochasticBagSpanTests(unittest.TestCase):
    def test_dim_one_bag_stays_tiny(self):
        pairs = ["A-B", "B-C", "C-D", "D-E", "E-F", "F-G", "G-H"]
        selected = _stochastic_bag_sample(pairs, 1, random.Random(0), min_pairs=5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(pid in pairs for pid in selected))

    def test_width_scales_bag_target(self):
        pairs = [f"A{i}-B{i}" for i in range(40)]
        selected = _stochastic_bag_sample(pairs, 4, random.Random(0), min_pairs=5)
        self.assertEqual(len(selected), 20)

    def test_default_span_bars_is_50(self):
        span = _stochastic_span_bars(1000, 4, HighRNG(), span_bars=50)
        self.assertEqual(span, 50)

    def test_span_grows_commensurately(self):
        self.assertEqual(_stochastic_span_bars(1000, 16, HighRNG(), span_bars=50), 100)
        self.assertEqual(_stochastic_span_bars(1000, 64, HighRNG(), span_bars=50), 200)

    def test_span_override_works(self):
        self.assertEqual(_stochastic_span_bars(1000, 4, HighRNG(), span_bars=25), 25)

    def test_non_autoresearch_bag_cap_is_50(self):
        self.assertEqual(_stochastic_bag_limit(64), 50)

    def test_non_autoresearch_bag_cap_override(self):
        self.assertEqual(_stochastic_bag_limit(64, cap=12), 12)


if __name__ == "__main__":
    unittest.main()
