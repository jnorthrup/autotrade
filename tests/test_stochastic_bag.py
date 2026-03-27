import random
from graph_showdown import (
    _BAG_SIZE_SCALE,
    _stochastic_bag_sample,
    _build_pair_adjacency,
    _select_related_pairs,
    _make_trial_graph,
)

def test_bag_size_scale_mapping():
    assert _BAG_SIZE_SCALE == {4: 5, 16: 20, 64: 40, 256: 80}

def test_build_pair_adjacency():
    pairs = ["BTC-USDT", "ETH-USDT", "BTC-ETH"]
    adj = _build_pair_adjacency(pairs)
    # Expected adjacency:
    # BTC: ["BTC-USDT", "BTC-ETH"]
    # USDT: ["BTC-USDT", "ETH-USDT"]
    # ETH: ["ETH-USDT", "BTC-ETH"]
    assert set(adj["BTC"]) == {"BTC-USDT", "BTC-ETH"}
    assert set(adj["USDT"]) == {"BTC-USDT", "ETH-USDT"}
    assert set(adj["ETH"]) == {"ETH-USDT", "BTC-ETH"}
    # Ensure no duplicates
    for v in adj.values():
        assert len(v) == len(set(v))

def test_select_related_pairs():
    pairs = ["A-B", "B-C", "C-D", "E-F"]
    adj = _build_pair_adjacency(pairs)
    rng = random.Random(42)
    # Select 2 pairs: should be connected
    selected = _select_related_pairs(pairs, adj, 2, rng)
    assert len(selected) == 2
    # Check that selected pairs share at least one currency? Not necessarily, but the algorithm tries to pick related.
    # We'll just test that it returns a subset.
    assert set(selected).issubset(set(pairs))
    # Select more than available: should return all
    selected_all = _select_related_pairs(pairs, adj, 10, rng)
    assert set(selected_all) == set(pairs)

def test_stochastic_bag_sample():
    # Use a fixed set of pairs
    pairs = ["BTC-USDT", "ETH-USDT", "BTC-ETH", "ADA-USDT", "SOL-USDT"]
    rng = random.Random(123)
    # Test model size 4 -> bag size 5 (but we have only 5 pairs, so should return all)
    selected = _stochastic_bag_sample(pairs, 4, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 5
    assert set(selected) == set(pairs)
    # Test model size 16 -> bag size 20, but we have only 5, so should return all
    selected = _stochastic_bag_sample(pairs, 16, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 5
    # Test model size 64 -> bag size 40 -> all
    selected = _stochastic_bag_sample(pairs, 64, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 5
    # Test model size 256 -> bag size 80 -> all
    selected = _stochastic_bag_sample(pairs, 256, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 5
    # Now test with more pairs to see scaling
    many_pairs = [f"COIN{i}-USDT" for i in range(20)] + [f"COIN{i}-COIN{j}" for i in range(5) for j in range(i+1,5)]
    rng = random.Random(456)
    # Model size 4 -> bag size 5
    selected = _stochastic_bag_sample(many_pairs, 4, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 5
    # Model size 16 -> bag size 20
    selected = _stochastic_bag_sample(many_pairs, 16, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 20
    # Model size 64 -> bag size 40, but we have only 20 + 10 = 30 pairs? Let's compute: 20 USDT pairs + 10 coin-coin pairs = 30.
    # So should return all 30 because bag size 40 > 30.
    selected = _stochastic_bag_sample(many_pairs, 64, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 30
    # Model size 256 -> bag size 80 -> all 30
    selected = _stochastic_bag_sample(many_pairs, 256, rng, min_pairs=5, max_pairs=None)
    assert len(selected) == 30
    # Test clamping with max_pairs
    selected = _stochastic_bag_sample(many_pairs, 64, rng, min_pairs=5, max_pairs=15)
    assert len(selected) == 15
    # Test with empty list
    assert _stochastic_bag_sample([], 4, rng) == []

# Mock classes for _make_trial_graph test
class MockEdgeState:
    pass
class MockNodeState:
    pass
class MockCoinGraph:
    def __init__(self, fee_rate=0.0):
        self.fee_rate = fee_rate
        self.all_pairs = []
        self.edges = {}  # (base, quote) -> some edge data
        self.edge_state = {}
        self.nodes = set()
        self.node_state = {}
        self.common_timestamps = list(range(100))  # 100 timestamps

def test_make_trial_graph():
    from coin_graph import EdgeState, NodeState
    full_graph = MockCoinGraph()
    full_graph.all_pairs = ["BTC-USDT", "ETH-USDT"]
    full_graph.edges = {
        ("BTC", "USDT"): "edge_data_BTC_USDT",
        ("USDT", "BTC"): "edge_data_USDT_BTC",
        ("ETH", "USDT"): "edge_data_ETH_USDT",
        ("USDT", "ETH"): "edge_data_USDT_ETH",
    }
    full_graph.common_timestamps = list(range(50))
    selected_pairs = ["BTC-USDT"]
    start_bar = 10
    end_bar = 20
    trial = _make_trial_graph(full_graph, selected_pairs, start_bar, end_bar)
    assert trial.fee_rate == full_graph.fee_rate
    assert trial.all_pairs == selected_pairs
    assert trial.edges == {
        ("BTC", "USDT"): "edge_data_BTC_USDT",
        ("USDT", "BTC"): "edge_data_USDT_BTC",
    }
    # Check that edge_state contains EdgeState instances
    assert isinstance(trial.edge_state[("BTC", "USDT")], EdgeState)
    assert isinstance(trial.edge_state[("USDT", "BTC")], EdgeState)
    # Check that node_state contains NodeState instances
    assert isinstance(trial.node_state["BTC"], NodeState)
    assert isinstance(trial.node_state["USDT"], NodeState)
    assert trial.nodes == {"BTC", "USDT"}
    assert trial.common_timestamps == list(range(10, 20))

if __name__ == "__main__":
    # Run tests manually if needed
    test_bag_size_scale_mapping()
    test_build_pair_adjacency()
    test_select_related_pairs()
    test_stochastic_bag_sample()
    test_make_trial_graph()
    print("All tests passed")