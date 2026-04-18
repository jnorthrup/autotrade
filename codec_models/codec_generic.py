"""
Generic Codec Implementation
=============================

A generic codec that can be used for any of the 24 codec types.
This serves as a template and fallback implementation.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base_codec import BaseCodec

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class GenericCodec(BaseCodec):
    """
    Generic codec implementation that can handle any codec type
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get('name', 'generic_codec')
        self.codec_id = config.get('codec_id', 0)
        
        # Create MLX model if available
        if HAS_MLX:
            self.model = self._create_mlx_model()
            print(f"✅ {self.name}: MLX model initialized")
        else:
            self.model = None
            print(f"⚠️  {self.name}: Using NumPy fallback")
        
        # Performance tracking
        self.recent_sharpe = 0.0
        self.weight = 1.0  # Dirichlet weight
    
    def _create_mlx_model(self):
        """Create MLX neural network"""
        return nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # [confidence, direction]
        )
    
    def forward(self, 
                market_data: Dict[str, Any],
                features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal using MLX model or simple logic
        """
        if HAS_MLX and self.model is not None:
            # Ensure features are 15-dimensional
            if len(features) < 15:
                padded = np.zeros(15, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            elif len(features) > 15:
                features = features[:15]
            
            try:
                # MLX forward pass
                features_mx = mx.array(features.reshape(1, -1))
                output = self.model(features_mx)
                
                confidence = float(output[0, 0])
                direction = float(output[0, 1])
            except Exception as e:
                print(f"⚠️  {self.name}: MLX forward failed: {e}")
                confidence, direction = self._simple_logic(market_data, features)
        else:
            # NumPy fallback
            confidence, direction = self._simple_logic(market_data, features)
        
        # Validate and normalize
        confidence, direction = self.validate_signal(confidence, direction)
        
        return confidence, direction
    
    def _simple_logic(self, 
                     market_data: Dict[str, Any], 
                     features: np.ndarray) -> Tuple[float, float]:
        """
        Simple fallback logic based on market data
        
        Uses:
        - LOB imbalance (positive = more buyers = potential price up)
        - Volume momentum
        - Basic TA signals
        """
        lob_imbalance = market_data.get('lob_imbalance', 0.0)
        price = market_data.get('price', 70000.0)
        volume = market_data.get('volume', 100000.0)
        
        # Simple momentum indicator
        momentum = 0.0
        if len(features) > 10:
            momentum = features[10]  # momentum_20
        
        # Combine signals
        signals = []
        
        # LOB imbalance (40% weight)
        if abs(lob_imbalance) > 0.05:
            signals.append(lob_imbalance * 0.4)
        
        # Momentum (30% weight)
        if abs(momentum) > 0.01:
            signals.append(momentum * 0.3)
        
        # Volume trend (30% weight)
        if volume > 1000000:  # High volume
            signals.append(0.1 if lob_imbalance > 0 else -0.1)
        
        if not signals:
            return 0.0, 0.0
        
        direction = np.mean(signals)
        confidence = abs(direction) + 0.2  # Base confidence
        
        return confidence, direction
    
    def test_time_adapter(self, 
                         batch_data: Dict[str, Any],
                         learning_rate: float = 1e-3) -> None:
        """
        Online fine-tuning for MLX model
        """
        if not HAS_MLX or self.model is None:
            return
        
        if 'inputs' not in batch_data or 'targets' not in batch_data:
            return
        
        try:
            # Prepare data
            inputs_mx = mx.array(batch_data['inputs'].astype(np.float32))
            targets_mx = mx.array(batch_data['targets'].astype(np.float32))
            
            # Define loss function
            def loss_fn(params):
                predictions = self.model.apply(params, inputs_mx)
                return mx.mean((predictions - targets_mx) ** 2)
            
            # Create optimizer
            optimizer = optim.Adam(learning_rate=learning_rate)
            
            # Get gradients and update
            loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
            optimizer.update(self.model, grads)
            
            print(f"✅ {self.name}: Online update, loss: {float(loss):.4f}")
        except Exception as e:
            print(f"⚠️  {self.name}: Online update failed: {e}")


    def online_adapter(self, *args, **kwargs) -> None:
        pass


# Factory function
def create_codec(config: Dict[str, Any] = None):
    """Create generic codec instance"""
    if config is None:
        config = {}
    return GenericCodec(config)

