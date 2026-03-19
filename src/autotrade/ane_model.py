import ctypes
import os
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# --- ANE C-Bridge ---

class ANEBridge:
    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = str(Path(__file__).parent / "ANE" / "bridge" / "libane_bridge.dylib")
        
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
    ANE-optimized reasoning block using 1D Convolutions to simulate message passing.
    """
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bridge = get_bridge()
        self.handle = None
        self.is_compiled = False
        
        self.mil_text = self._generate_mil(hidden_size)

    def _generate_mil(self, hidden_size):
        # ANE is built for convolutions. We use a Conv1D bottleneck [1, C, 1, 1]
        # This simulates a high-performance linear layer interaction.
        return f"""
        main(x: fp16[1, {hidden_size}, 1, 1], w1: fp16[{hidden_size}, {hidden_size}, 1, 1], w2: fp16[{hidden_size}, {hidden_size}, 1, 1]) -> (fp16[1, {hidden_size}, 1, 1]) {{
          res1 = conv(x=x, weight=w1, strides=[1, 1], pad_type="valid", groups=1)
          act1 = relu(x=res1)
          res2 = conv(x=act1, weight=w2, strides=[1, 1], pad_type="valid", groups=1)
          return res2
        }}
        """.strip()

    def compile(self, w1: mx.array, w2: mx.array):
        # Build multi-weight blob for ANE
        w1_np = np.array(w1).astype(np.float32)
        w2_np = np.array(w2).astype(np.float32)
        
        # For simplicity in this seed, we concatenate weights or pack them.
        # Real impl uses 'ane_bridge_compile_multi_weights'
        # Here we use a single packed blob for the two matrices.
        packed_w = np.concatenate([w1_np.flatten(), w2_np.flatten()])
        
        out_len = ctypes.c_size_t()
        blob_ptr = self.bridge.lib.ane_bridge_build_weight_blob(
            packed_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            2, self.hidden_size * self.hidden_size, ctypes.byref(out_len)
        )
        
        in_size = (ctypes.c_size_t * 1)(self.hidden_size * 2)
        out_size = (ctypes.c_size_t * 1)(self.hidden_size * 2)
        
        self.handle = self.bridge.lib.ane_bridge_compile(
            self.mil_text.encode('utf-8'), len(self.mil_text),
            blob_ptr, out_len,
            1, in_size,
            1, out_size
        )
        self.bridge.lib.ane_bridge_free_blob(blob_ptr)
        self.is_compiled = True

    def __call__(self, x: mx.array) -> mx.array:
        if not self.is_compiled: return x
        x_np = np.array(x).astype(np.float16)
        self.bridge.lib.ane_bridge_write_input(self.handle, 0, x_np.ctypes.data, x_np.nbytes)
        if not self.bridge.lib.ane_bridge_eval(self.handle): return x
        out_np = np.empty_like(x_np)
        self.bridge.lib.ane_bridge_read_output(self.handle, 0, out_np.ctypes.data, out_np.nbytes)
        return mx.array(out_np)

# --- Integrated Model ---

class HRM_ANE(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, H_cycles=2, L_cycles=3, x_pixels=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_pixels = x_pixels
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        self.encoder = nn.Linear(x_pixels + 4, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, x_pixels)
        )
        
        # Weights for the ANE block (managed by MLX)
        self.ane_w1 = mx.random.normal((hidden_size, hidden_size)) * 0.02
        self.ane_w2 = mx.random.normal((hidden_size, hidden_size)) * 0.02
        
        self.ane_block = ANERecurrentBlock(hidden_size, num_heads)
        self._compiled = False

    def __call__(self, x):
        if not self._compiled:
            self.ane_block.compile(self.ane_w1, self.ane_w2)
            self._compiled = True
            
        B, n_edges, _ = x.shape
        x_in = self.encoder(x)
        
        # HRM Thinking Cycles
        z_H = mx.zeros((B, n_edges, self.hidden_size))
        z_L = mx.zeros((B, n_edges, self.hidden_size))
        
        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                # Run ANE block per edge (simulated iteration)
                # In full impl, we'd batch all edges into the ANE spatial dim
                z_L = self.ane_block(z_L + x_in + z_H)
            z_H = self.ane_block(z_H + z_L)
            
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
