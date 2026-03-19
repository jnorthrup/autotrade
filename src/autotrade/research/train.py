"""
Autoresearch pretraining script. Single-device, single-file.
Apple Silicon MLX + ANE port with HRM (Recurrent) Architecture.
Exploits entropy via Symmetric Fisheye Prediction.
"""

import gc
import math
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from .prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader
from ..ane_model import HRM_ANE

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# --- Mirroring Physics: Symmetric Multi-Horizon Loss ---

class HyperbolicMultiHorizonLoss:
    """
    Entropy-exploiting loss for staged token prediction.
    Rewards capturing trend complexity across symmetric horizons.
    """
    def __init__(self, power=2.0, n_horizons=5):
        self.power = power
        self.n_horizons = n_horizons
        # Symmetric weights: longer horizons rewarded more
        self.weights = mx.array([1.0, 1.5, 2.0, 3.0, 5.0])[:n_horizons]

    def __call__(self, logits_staged, targets):
        # logits_staged: List of n_horizons tensors, each (B, T, V)
        total_loss = 0.0
        for h in range(self.n_horizons):
            logits = logits_staged[h]
            # Predict t+1+h
            if h == 0:
                h_targets = targets
                h_logits = logits
            else:
                h_targets = targets[:, h:]
                h_logits = logits[:, :-h]
            
            valid = h_targets != -1
            targets_safe = mx.where(valid, h_targets, 0)
            ce = nn.losses.cross_entropy(h_logits, targets_safe, reduction="none")
            loss_h = mx.mean(ce * valid)
            
            # Intensity Reward: capturing high-entropy token transitions
            total_loss = total_loss + loss_h * self.weights[h]
            
        return total_loss / mx.sum(self.weights)

# --- Model Evolution: HRM-ANE GPT ---

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    H_cycles: int = 2
    L_cycles: int = 3

class HRM_GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Recurrent HRM Core (Offloaded to ANE)
        self.hrm = HRM_ANE(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            x_pixels=64 # Coherent resolution
        )
        
        # MTP Heads for Symmetric Staged Output
        self.lm_heads = [nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(5)]

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        
        # Inject embeddings into the ANE reasoning core
        # We dummy-pad to match the 4-dim meta expected by the core
        meta = mx.zeros((B, T, 4))
        z = self.hrm.encoder(mx.concatenate([x, meta], axis=-1))
        
        # Recurrent Thinking Cycles (Thinking bridges the resolution gap)
        # In this GPT impl, we treat the sequence dimension like the graph edge dimension
        z_out = self.hrm.ane_block(z)
        
        # Decode into 5 staged futures
        logits_staged = [head(z_out) for head in self.lm_heads]
        
        # Apply tanh softcap to prevent entropy collapse
        logits_staged = [15.0 * mx.tanh(l / 15.0) for l in logits_staged]

        if targets is None:
            return logits_staged[0]
        return logits_staged

    def init_weights(self):
        scale = 3**0.5 * self.config.n_embd**-0.5
        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        for head in self.lm_heads:
            head.weight = (mx.random.normal(head.weight.shape) * 0.001).astype(mx.bfloat16)

# --- Training Infrastructure (Sophisticated MLX Harness) ---

class AdamW:
    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, scalar_lr):
        self.param_config = {}
        self.adam_state = {}
        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            lr = matrix_lr
            if "wte" in path or "hrm" in path: lr = embedding_lr * dmodel_lr_scale
            elif "lm_heads" in path: lr = unembedding_lr * dmodel_lr_scale
            
            self.param_config[path] = {
                "lr": lr, "betas": adam_betas, "eps": 1e-10, "weight_decay": weight_decay if "weight" in path else 0.0
            }
        self.initial_lrs = {path: config["lr"] for path, config in self.param_config.items()}

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config: continue
            config, param = self.param_config[path], flat_params[path]
            
            if path not in self.adam_state:
                self.adam_state[path] = {"m": mx.zeros_like(param), "v": mx.zeros_like(param), "t": 0}
            
            state = self.adam_state[path]
            state["t"] += 1
            m, v, t = state["m"], state["v"], state["t"]
            beta1, beta2 = config["betas"]
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            state["m"], state["v"] = m, v
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            new_p = param * (1 - config["lr"] * config["weight_decay"])
            new_p = new_p - config["lr"] * m_hat / (mx.sqrt(v_hat) + config["eps"])
            
            # In-place update using MLX tree_map or manual path setting
            self._set_path_value(model, path, new_p)

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if "[" in part: # handle heads[0] etc
                name, idx = part.split("[")
                obj = getattr(obj, name)[int(idx[:-1])]
            elif isinstance(obj, dict): obj = obj[part]
            else: obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

# --- Hyperparameters ---

DEPTH = 12
ASPECT_RATIO = 64
HEAD_DIM = 128
TOTAL_BATCH_SIZE = 2**16
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
TIME_BUDGET = 300 # 5 minutes

def main():
    t_start = time.time()
    mx.random.seed(42)
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    train_loader = make_dataloader(tokenizer, 16, MAX_SEQ_LEN, "train")
    
    config = GPTConfig(vocab_size=vocab_size, n_layer=DEPTH)
    model = HRM_GPT(config)
    model.init_weights()
    
    optimizer = AdamW(model, UNEMBEDDING_LR, EMBEDDING_LR, MATRIX_LR, WEIGHT_DECAY, ADAM_BETAS, SCALAR_LR)
    loss_fn = HyperbolicMultiHorizonLoss(n_horizons=5)
    
    def loss_fn_wrap(model, x, y):
        return loss_fn(model(x, targets=y), y)
    
    loss_grad_fn = nn.value_and_grad(model, loss_fn_wrap)

    print(f"Symmetric HRM-ANE Research Seed Started. Budget: {TIME_BUDGET}s")
    step = 0
    while time.time() - t_start < TIME_BUDGET:
        x, y, epoch = next(train_loader)
        loss, grads = loss_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        if step % 10 == 0: print(f"\rStep {step} | Loss: {loss.item():.4f}", end="")
        step += 1

    val_bpb = evaluate_bpb(model, tokenizer, 128)
    print(f"\nFinal val_bpb: {val_bpb:.6f}")

if __name__ == "__main__":
    main()
