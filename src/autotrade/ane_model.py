import ctypes
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Optional, Tuple, Any


@contextmanager
def _suppress_stderr():
    devnull = open(os.devnull, 'w')
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        devnull.close()

# --- State Dataclasses ---

@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)

@dataclass
class EdgeState:
    base: str
    quote: str
    curvature: float = 2.0
    y_depth: int = 200
    x_pixels: int = 20

# --- ANE C-Bridge ---

class ANEBridge:
    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = str(Path(__file__).parent.parent.parent / "ANE" / "bridge" / "libane_bridge.dylib")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"ANE Bridge library not found at {lib_path}. Run 'make' in bridge directory.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.ane_bridge_init.restype = ctypes.c_int
        if self.lib.ane_bridge_init() != 0:
            raise RuntimeError("Failed to initialize ANE bridge")
            
        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p
        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_eval.restype = ctypes.c_bool
        self.lib.ane_bridge_write_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_read_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_build_weight_blob.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_ubyte)
        self._has_free_blob = hasattr(self.lib, 'ane_bridge_free_blob')
        if self._has_free_blob:
            self.lib.ane_bridge_free_blob.argtypes = [ctypes.c_void_p]

_bridge = None
def get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = ANEBridge()
    return _bridge

# --- ANE-Optimized Recurrent Block ---

class ANERecurrentBlock:
    """
    ANE-optimized reasoning block with MLX fallback.
    Uses ANE when available, falls back to pure MLX otherwise.
    """
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bridge = None
        self.handle = None
        self.is_compiled = False
        self._use_ane = False
        
        self._w1: Optional[mx.array] = None
        self._w2: Optional[mx.array] = None
        
        try:
            self.bridge = get_bridge()
            self.mil_text = self._generate_mil(hidden_size)
            self._use_ane = True
        except Exception:
            pass

    def _generate_mil(self, hidden_size):
        return f"""
        main(x: fp16[1, {hidden_size}, 1, 1], w1: fp16[{hidden_size}, {hidden_size}, 1, 1], w2: fp16[{hidden_size}, {hidden_size}, 1, 1]) -> (fp16[1, {hidden_size}, 1, 1]) {{
          res1 = conv(x=x, weight=w1, strides=[1, 1], pad_type="valid", groups=1)
          act1 = relu(x=res1)
          res2 = conv(x=act1, weight=w2, strides=[1, 1], pad_type="valid", groups=1)
          return res2
        }}
        """.strip()

    def compile(self, w1: mx.array, w2: mx.array):
        self._w1 = w1
        self._w2 = w2
        
        if not self._use_ane or self.bridge is None:
            return
        
        try:
            w1_np = np.array(w1).astype(np.float32)
            w2_np = np.array(w2).astype(np.float32)
            
            packed_w = np.concatenate([w1_np.flatten(), w2_np.flatten()])
            
            out_len = ctypes.c_size_t()
            blob_ptr = self.bridge.lib.ane_bridge_build_weight_blob(
                packed_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                2, self.hidden_size * self.hidden_size, ctypes.byref(out_len)
            )
            
            in_size = (ctypes.c_size_t * 1)(self.hidden_size * 2)
            out_size = (ctypes.c_size_t * 1)(self.hidden_size * 2)
            
            with _suppress_stderr():
                self.handle = self.bridge.lib.ane_bridge_compile(
                    self.mil_text.encode('utf-8'), len(self.mil_text),
                    blob_ptr, out_len,
                    1, in_size,
                    1, out_size
                )
            if self.bridge._has_free_blob:
                self.bridge.lib.ane_bridge_free_blob(blob_ptr)
            self.is_compiled = self.handle is not None
        except Exception:
            self.is_compiled = False

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_compiled and self.handle is not None:
            try:
                x_np = np.array(x).astype(np.float16)
                self.bridge.lib.ane_bridge_write_input(self.handle, 0, x_np.ctypes.data, x_np.nbytes)
                if self.bridge.lib.ane_bridge_eval(self.handle):
                    out_np = np.empty_like(x_np)
                    self.bridge.lib.ane_bridge_read_output(self.handle, 0, out_np.ctypes.data, out_np.nbytes)
                    return mx.array(out_np)
            except Exception:
                pass
        
        if self._w1 is not None and self._w2 is not None:
            h = mx.maximum(0, x @ self._w1.T)
            return h @ self._w2.T
        return x

# --- Integrated Model ---

class HRMBlock(nn.Module):
    """Simple differentiable block for HRM cycles."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.fc1(x))
        return self.fc2(h) + x  # residual

class HRM_ANE(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, H_cycles=2, L_cycles=3, x_pixels=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_pixels = x_pixels
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        input_dim = x_pixels + 4 + 16  # fisheye + metadata + peer features
        self.encoder = nn.Linear(input_dim, hidden_size)
        
        # HRM blocks - properly tracked as parameters
        self.hrm_low = HRMBlock(hidden_size)
        self.hrm_high = HRMBlock(hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1)  # output single velocity per edge
        )

    def __call__(self, x):
        B, n_edges, _ = x.shape
        x_in = self.encoder(x)
        
        # HRM Thinking Cycles
        z_H = mx.zeros((B, n_edges, self.hidden_size))
        z_L = mx.zeros((B, n_edges, self.hidden_size))
        
        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z_L = self.hrm_low(z_L + x_in + z_H)
            z_H = self.hrm_high(z_H + z_L)
            
        return self.decoder(z_H)

# --- Perspective Mirroring ---

def fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float) -> List[int]:
    boundaries, prev = [], 0
    for i in range(x_pixels):
        t = (i + 1) / x_pixels
        b = max(int(y_depth * (t ** curvature)), prev + 1)
        boundaries.append(b)
        prev = b
    return boundaries

def fisheye_sample(candles: np.ndarray, boundaries: List[int]) -> List[float]:
    results, prev = [], 0
    for b in boundaries:
        bucket = candles[max(0, len(candles)-b):max(0, len(candles)-prev)]
        if len(bucket) == 0: results.append(0.0)
        else:
            m = np.mean(bucket)
            results.append((candles[-1] - m) / m if m != 0 else 0.0)
        prev = b
    return results


class AccelModel:
    """
    ANE-accelerated AccelModel matching accel_model.py API.
    Uses MLX for Apple Silicon optimization with optional ANE bridge.
    """
    def __init__(self, n_edges: int = 0, sequence_length: int = 8, learning_rate: float = 0.001,
                 hidden_size: int = 128, num_heads: int = 4, H_cycles: int = 2, L_cycles: int = 3,
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.hidden_size = hidden_size
        
        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.edge_list: List[Tuple[int, int]] = []
        
        self._default_curvature = curvature
        self._default_y_depth = y_depth
        self._default_x_pixels = x_pixels
        
        self._edge_states: Dict[Tuple[str, str], EdgeState] = {}
        
        self.model: Optional[HRM_ANE] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)
        
        self._bar_count = 0
        self._high_level_update_freq = 10
        self._max_history = max(y_depth + 100, 500)
    
    def register_edges(self, edges: List[Tuple[str, str]]):
        """Build graph structure and initialize fisheye parameters."""
        self.edge_names = edges
        self.n_edges = len(edges)
        
        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)
        
        self.node_names = sorted(list(node_set))
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        
        self.edge_list = []
        for base, quote in edges:
            base_idx = self.node_to_idx[base]
            quote_idx = self.node_to_idx[quote]
            self.edge_list.append((base_idx, quote_idx))
            self.edge_to_idx[(base, quote)] = len(self.edge_list) - 1
        
        for edge in edges:
            self._edge_states[edge] = EdgeState(
                base=edge[0], quote=edge[1],
                curvature=self._default_curvature,
                y_depth=self._default_y_depth,
                x_pixels=self._default_x_pixels
            )
        
        self.model = HRM_ANE(
            hidden_size=self.hidden_size,
            x_pixels=self.x_pixels,
            H_cycles=2, L_cycles=3
        )
        self.optimizer = optim.Adam(learning_rate=self._lr)
    
    def _build_input_tensor(self, graph) -> mx.array:
        """Build per-edge feature matrix: (1, n_edges, x_pixels + 4 + 16)"""
        edge_features = []
        
        peer_nodes = []
        if hasattr(graph, 'node_state'):
            sorted_nodes = sorted(graph.node_state.items(), key=lambda x: abs(x[1].height), reverse=True)
            peer_nodes = [n for n, _ in sorted_nodes[:4]]
        
        while len(peer_nodes) < 4:
            peer_nodes.append(None)
        
        for edge in self.edge_names:
            base, quote = edge
            state = self._edge_states.get(edge)
            
            curvature = float(np.clip(state.curvature, 0.1, 10.0)) if state else self._default_curvature
            y_depth = int(np.clip(state.y_depth if state else self._default_y_depth, 10, 1000))
            x_pixels = int(np.clip(state.x_pixels if state else self._default_x_pixels, 5, 50))
            
            boundaries = fisheye_boundaries(y_depth, x_pixels, curvature)
            closes = self._close_buffer.get(edge, [])
            
            if len(closes) < 2:
                fisheye_values = [0.0] * self.x_pixels
            else:
                closes_array = np.array(closes[-y_depth:])
                fisheye_values = fisheye_sample(closes_array, boundaries)
            
            fisheye_values = fisheye_values[:self.x_pixels]
            while len(fisheye_values) < self.x_pixels:
                fisheye_values.append(0.0)
            
            feat = fisheye_values + [
                curvature,
                y_depth / 1000.0,
                x_pixels / 100.0,
                getattr(graph, 'fee_rate', 0.001)
            ]
            
            for peer in peer_nodes:
                if peer and hasattr(graph, 'node_state') and peer in graph.node_state:
                    ns = graph.node_state[peer]
                    is_involved = 1.0 if peer in [base, quote] else 0.0
                    feat += [ns.height, ns.net_inflow, ns.time_at_north / 100.0, is_involved]
                else:
                    feat += [0.0, 0.0, 0.0, 0.0]
            
            edge_features.append(feat)
        
        return mx.array(np.array(edge_features, dtype=np.float32)).reshape(1, len(edge_features), -1)
    
    def high_level_plan(self, graph) -> Dict[str, str]:
        """Infer high-level direction per currency from graph structure."""
        self._bar_count += 1
        
        for node in self.node_names:
            if node not in self.hrm_state:
                self.hrm_state[node] = HRMState(currency=node)
        
        dir_map = {}
        for node in self.node_names:
            if hasattr(graph, 'node_state') and node in graph.node_state:
                height = graph.node_state[node].height
                if height > 0.0001:
                    dir_map[node] = "north"
                elif height < -0.0001:
                    dir_map[node] = "south"
                else:
                    dir_map[node] = "neutral"
            else:
                dir_map[node] = "neutral"
            
            self.hrm_state[node].high_level_direction = dir_map[node]
            self._direction_buffer[node].append(dir_map[node])
            if len(self._direction_buffer[node]) > self.sequence_length:
                self._direction_buffer[node].pop(0)
        
        return {c: hs.high_level_direction for c, hs in self.hrm_state.items()}
    
    def predict(self, graph) -> Dict[Tuple[str, str], float]:
        """Predict per-edge velocity using HRM_ANE model."""
        if not self.edge_names or not self.model:
            return {}
        
        edge_features = self._build_input_tensor(graph)
        
        velocities = self.model(edge_features)
        preds = np.array(velocities).flatten().tolist()
        
        if isinstance(preds, float):
            preds = [preds]
        
        predictions = {}
        for i, edge in enumerate(self.edge_names):
            predictions[edge] = preds[i] if i < len(preds) else 0.0
        
        return predictions
    
    def update_prices(self, graph, bar_idx: int):
        """Update close price buffer from graph for fisheye sampling."""
        if bar_idx < 0:
            return
        
        if not hasattr(graph, 'edges'):
            return
        
        for edge in self.edge_names:
            if edge in graph.edges:
                df = graph.edges[edge]
                if bar_idx < len(df):
                    close = float(df['close'].iloc[bar_idx])
                    self._close_buffer[edge].append(close)
                    if len(self._close_buffer[edge]) > self._max_history:
                        self._close_buffer[edge] = self._close_buffer[edge][-self._max_history:]
    
    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1) -> Optional[float]:
        """Update model with actual velocities and calculate HRM reliability."""
        if not self.edge_names or not actual_accels or not self.model:
            return None
        
        self.update_prices(graph, bar_idx)
        
        edge_features = self._build_input_tensor(graph)
        target = mx.array([actual_accels.get(edge, 0.0) for edge in self.edge_names])
        
        def loss_fn(model):
            velocities = model(edge_features)
            preds = velocities.reshape(-1)
            return mx.mean(mx.square(preds - target))
        
        loss, grad = mx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grad)
        
        velocities = self.model(edge_features)
        preds = np.array(velocities).flatten().tolist()
        
        for i, edge in enumerate(self.edge_names):
            base, quote = edge
            actual = actual_accels.get(edge, 0.0)
            pred = preds[i]
            
            if base in self.hrm_state:
                self.hrm_state[base].predictions.append(pred)
                self.hrm_state[base].actuals.append(actual)
                if len(self.hrm_state[base].predictions) > 100:
                    self.hrm_state[base].predictions.pop(0)
                    self.hrm_state[base].actuals.pop(0)
                
                if len(self.hrm_state[base].predictions) > 10:
                    corr = np.corrcoef(self.hrm_state[base].predictions, self.hrm_state[base].actuals)[0, 1]
                    self.hrm_state[base].low_level_accuracy = 0.0 if np.isnan(corr) else corr
        
        return float(loss)
    
    def get_hrm_stats(self) -> List[Dict]:
        """Return HRM statistics for leaderboard."""
        stats = []
        for currency, hs in self.hrm_state.items():
            stats.append({
                'currency': currency,
                'high_level_direction': hs.high_level_direction,
                'low_level_accuracy': hs.low_level_accuracy
            })
        return stats
