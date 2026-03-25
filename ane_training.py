"""
ANE Training Integration for HRM

Interface between HierarchicalReasoningModel and the ANE training system
located in ./ANE/training. Uses subprocess to run compiled Objective-C
training programs that execute on Apple Neural Engine hardware.

The ANE training system uses reverse-engineered private APIs to run
backpropagation directly on ANE hardware via MIL (Model Intermediate Language).
"""
import os
import subprocess
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from hrm_model import HRMEdgePredictor, HierarchicalReasoningModel


class ANECheckpointFormat:
    """
    Binary checkpoint format compatible with ANE training code.

    Format:
    - Header: magic (4 bytes) + version (4 bytes) + num_params (4 bytes)
    - Per param: name_len (4 bytes) + name_bytes + shape_len (4 bytes) +
                 shape_data + dtype (4 bytes) + data_bytes
    """

    MAGIC = b'ANECKPT'
    VERSION = 1

    @staticmethod
    def save(params: Dict[str, torch.Tensor], path: str):
        """Save PyTorch state_dict to ANE checkpoint format."""
        with open(path, 'wb') as f:
            # Header
            f.write(ANECheckpointFormat.MAGIC)
            f.write(struct.pack('<I', ANECheckpointFormat.VERSION))
            f.write(struct.pack('<I', len(params)))

            # Write each parameter
            for name, tensor in params.items():
                # Name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)

                # Shape
                f.write(struct.pack('<I', len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack('<I', dim))

                # Dtype (1=float32, 2=float16, 3=bfloat16)
                if tensor.dtype == torch.float32:
                    dtype = 1
                elif tensor.dtype == torch.float16:
                    dtype = 2
                elif tensor.dtype == torch.bfloat16:
                    # Convert bfloat16 to float32 for storage
                    tensor = tensor.float()
                    dtype = 1
                else:
                    raise ValueError(f"Unsupported dtype: {tensor.dtype}")
                f.write(struct.pack('<I', dtype))

                # Data
                data = tensor.detach().cpu().numpy().tobytes()
                f.write(struct.pack('<I', len(data)))
                f.write(data)

    @staticmethod
    def load(path: str) -> Dict[str, torch.Tensor]:
        """Load ANE checkpoint to PyTorch state_dict."""
        params = {}

        with open(path, 'rb') as f:
            # Header
            magic = f.read(8)
            if magic != ANECheckpointFormat.MAGIC:
                raise ValueError(f"Invalid checkpoint magic: {magic}")

            version = struct.unpack('<I', f.read(4))[0]
            if version != ANECheckpointFormat.VERSION:
                raise ValueError(f"Unsupported version: {version}")

            num_params = struct.unpack('<I', f.read(4))[0]

            # Read each parameter
            for _ in range(num_params):
                # Name
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')

                # Shape
                shape_len = struct.unpack('<I', f.read(4))[0]
                shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(shape_len))

                # Dtype
                dtype = struct.unpack('<I', f.read(4))[0]
                if dtype == 1:
                    np_dtype = np.float32
                elif dtype == 2:
                    np_dtype = np.float16
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")

                # Data
                data_len = struct.unpack('<I', f.read(4))[0]
                data_bytes = f.read(data_len)
                data = np.frombuffer(data_bytes, dtype=np_dtype)
                params[name] = torch.from_numpy(data).reshape(shape)

        return params


class ANETrainer:
    """
    Training interface for HRM using ANE hardware.

    Workflow:
    1. Export HRM weights to ANE checkpoint format
    2. Run compiled Objective-C training program (subprocess)
    3. Monitor training via dashboard IPC or stdout parsing
    4. Import updated weights back to HRM
    """

    def __init__(self, model: HierarchicalReasoningModel, ane_dir: str = "./ANE"):
        self.model = model
        self.ane_dir = Path(ane_dir)
        self.training_dir = self.ane_dir / "training"

        # Check ANE training system exists
        if not self.training_dir.exists():
            raise FileNotFoundError(f"ANE training directory not found: {self.training_dir}")

        # Training binary
        self.train_binary = self.training_dir / "train_large"
        if not self.train_binary.exists():
            raise FileNotFoundError(f"Training binary not found: {self.train_binary}")

        # Checkpoint paths
        self.ckpt_path = self.training_dir / "hrm_ckpt.bin"
        self.updated_ckpt = self.training_dir / "hrm_ckpt_updated.bin"

    def export_weights(self, path: Optional[str] = None):
        """Export HRM weights to ANE checkpoint format."""
        if path is None:
            path = self.ckpt_path

        if self.model._model is None:
            raise RuntimeError("Model not initialized. Call register_edges() first.")

        state_dict = self.model._model.state_dict()
        ANECheckpointFormat.save(state_dict, path)
        print(f"Exported {len(state_dict)} parameters to {path}")

    def import_weights(self, path: Optional[str] = None):
        """Import trained weights from ANE checkpoint."""
        if path is None:
            path = self.updated_ckpt

        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        params = ANECheckpointFormat.load(path)

        if self.model._model is None:
            raise RuntimeError("Model not initialized.")

        # Load state dict
        self.model._model.load_state_dict(params)
        print(f"Imported {len(params)} parameters from {path}")

    def train(
        self,
        steps: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        checkpoint_interval: int = 100,
    ) -> float:
        """
        Run ANE training subprocess.

        Args:
            steps: Number of training steps
            batch_size: Batch size
            learning_rate: Learning rate
            checkpoint_interval: Save checkpoint every N steps

        Returns:
            Final loss value
        """
        # Export current weights
        self.export_weights()

        # Build command
        cmd = [
            str(self.train_binary),
            f"--checkpoint={self.ckpt_path}",
            f"--output={self.updated_ckpt}",
            f"--steps={steps}",
            f"--batch_size={batch_size}",
            f"--lr={learning_rate}",
            f"--save_every={checkpoint_interval}",
        ]

        print(f"Launching ANE training: {' '.join(cmd)}")

        # Run training
        process = subprocess.Popen(
            cmd,
            cwd=str(self.training_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Monitor output
        final_loss = 0.0
        for line in process.stdout:
            print(f"[ANE] {line}", end='')
            # Parse loss from output (expected format: "Step N: loss=X.XXX")
            if "loss=" in line:
                try:
                    loss_str = line.split("loss=")[1].split()[0]
                    final_loss = float(loss_str)
                except (IndexError, ValueError):
                    pass

        # Wait for completion
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"ANE training failed with return code {return_code}")

        # Import updated weights
        self.import_weights()

        return final_loss


def enable_ane_training(model: HierarchicalReasoningModel) -> ANETrainer:
    """
    Enable ANE training for HRM model.

    Args:
        model: HierarchicalReasoningModel instance

    Returns:
        ANETrainer instance
    """
    return ANETrainer(model)


# ── Integration with graph_showdown ─────────────────────────────────────────────

def train_with_ane(
    model: HierarchicalReasoningModel,
    graph,
    start_bar: int = 0,
    end_bar: Optional[int] = None,
    steps: int = 1000,
) -> Tuple[float, int, List[float]]:
    """
    Training loop using ANE hardware.

    Exports fisheye features and targets to ANE format, runs training,
    and imports updated weights.

    Args:
        model: HRM model
        graph: Coin graph with training data
        start_bar: Starting bar index
        end_bar: Ending bar index (None = end of data)
        steps: Number of ANE training steps

    Returns:
        (total_loss, n_updates, loss_history)
    """
    if end_bar is None:
        end_bar = len(graph.common_timestamps)

    ane_trainer = ANETrainer(model)

    # Collect training data
    fisheye_data = []
    targets = []

    for bar_idx in range(start_bar, min(end_bar, len(graph.common_timestamps))):
        edge_accels, _, hit_ptt, hit_stop = graph.update(bar_idx)

        if not edge_accels:
            continue

        # Collect fisheye features for each edge
        for edge in model.edge_names:
            fisheye = model._get_fisheye(edge)
            fisheye_data.append(fisheye)

            # Get targets for this edge
            if edge in hit_ptt and edge in hit_stop:
                ptt_target = 1.0 if hit_ptt[edge] else 0.0
                stop_target = 1.0 if hit_stop[edge] else 0.0

                if hit_ptt[edge]:
                    frac_target = 1.0
                elif hit_stop[edge]:
                    frac_target = 0.0
                else:
                    frac_target = 0.5

                targets.append((frac_target, ptt_target, stop_target))

    # Export data to ANE format
    # TODO: Implement proper data export format for ANE training

    # Run ANE training
    loss = ane_trainer.train(steps=steps)

    return loss, steps, [loss]
