#!/usr/bin/env python3
"""
Generate initial HRM checkpoint file for ANE training.
Creates a binary checkpoint with random weights.
"""
import struct
import numpy as np
from pathlib import Path

def create_hrm_checkpoint(
    output_path: str = "hrm_ckpt.bin",
    h_dim: int = 4,
    z_dim: int = 4,
    num_edges: int = 32,
    fisheye_size: int = 128
):
    """Create initial HRM checkpoint with random weights."""

    # Checkpoint header
    magic = b"ANECKPT\0"
    version = 1
    num_params = 4  # W1, W2, Wq, Wk per edge

    with open(output_path, 'wb') as f:
        # Write header
        f.write(magic[:7])  # 7 bytes magic
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<I', num_params))

        # Write parameters for each edge
        for edge_idx in range(num_edges):
            # Parameter 1: W1 (fisheye projection)
            name = f"W1_edge{edge_idx}"
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = [fisheye_size, h_dim]
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            dtype = 0  # 0 = float32
            f.write(struct.pack('<I', dtype))

            # Random weights
            W1 = np.random.randn(fisheye_size, h_dim).astype(np.float32) * 0.02
            f.write(W1.tobytes())

            # Parameter 2: W2 (output projection)
            name = f"W2_edge{edge_idx}"
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = [z_dim, 3]  # 3 outputs: frac, ptt, stop
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            f.write(struct.pack('<I', dtype))
            W2 = np.random.randn(z_dim, 3).astype(np.float32) * 0.02
            f.write(W2.tobytes())

            # Parameter 3: Wq (query projection)
            name = f"Wq_edge{edge_idx}"
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = [h_dim, h_dim]
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            f.write(struct.pack('<I', dtype))
            Wq = np.random.randn(h_dim, h_dim).astype(np.float32) * 0.02
            f.write(Wq.tobytes())

            # Parameter 4: Wk (key projection)
            name = f"Wk_edge{edge_idx}"
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = [h_dim, h_dim]
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            f.write(struct.pack('<I', dtype))
            Wk = np.random.randn(h_dim, h_dim).astype(np.float32) * 0.02
            f.write(Wk.tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"✅ Created checkpoint: {output_path}")
    print(f"   Size: {file_size:,} bytes")
    print(f"   Edges: {num_edges}")
    print(f"   Architecture: h_dim={h_dim}, z_dim={z_dim}, fisheye={fisheye_size}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate initial HRM checkpoint")
    parser.add_argument("--output", default="hrm_ckpt.bin", help="Output checkpoint path")
    parser.add_argument("--h-dim", type=int, default=4, help="Hidden dimension")
    parser.add_argument("--z-dim", type=int, default=4, help="Z dimension")
    parser.add_argument("--num-edges", type=int, default=32, help="Number of edges")
    parser.add_argument("--fisheye-size", type=int, default=128, help="Fisheye feature size")

    args = parser.parse_args()

    create_hrm_checkpoint(
        output_path=args.output,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        num_edges=args.num_edges,
        fisheye_size=args.fisheye_size
    )
