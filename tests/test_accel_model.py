import numpy as np
import pytest
from accel_model import fisheye_sample

def test_fisheye_sample_empty_candles():
    """Test fisheye_sample with empty candles array."""
    candles = np.array([])
    boundaries = [1, 2]
    expected = [0.0, 0.0]
    result = fisheye_sample(candles, boundaries)
    assert result == expected

def test_fisheye_sample_zero_mean_handling():
    """Test handling of buckets with zero mean."""
    candles = np.array([-10.0, 10.0, 5.0])
    boundaries = [2]
    # Bucket 1: [-10.0, 10.0], mean=0.0
    expected = [0.0]
    result = fisheye_sample(candles, boundaries)
    assert result == expected

def test_fisheye_sample_boundary_clamping():
    """Test that boundaries exceeding candle length are clamped."""
    candles = np.array([10.0, 20.0])
    boundaries = [1, 5]
    # Bucket 1: [10.0], mean=10.0, close=20.0, result=(20-10)/10 = 1.0
    # Bucket 2: [20.0], mean=20.0, close=20.0, result=(20-20)/20 = 0.0
    expected = [1.0, 0.0]
    result = fisheye_sample(candles, boundaries)
    assert np.allclose(result, expected)

def test_fisheye_sample_normal_operation():
    """Test fisheye_sample with normal inputs."""
    candles = np.array([10.0, 20.0, 30.0])
    boundaries = [1, 3]
    # Bucket 1: [10.0], mean=10.0, close=30.0, result=(30-10)/10 = 2.0
    # Bucket 2: [20.0, 30.0], mean=25.0, close=30.0, result=(30-25)/25 = 0.2
    expected = [2.0, 0.2]
    result = fisheye_sample(candles, boundaries)
    assert np.allclose(result, expected)

def test_fisheye_sample_empty_boundaries():
    """Test fisheye_sample with empty boundaries list."""
    candles = np.array([1.0, 2.0])
    boundaries = []
    expected = []
    result = fisheye_sample(candles, boundaries)
    assert result == expected
