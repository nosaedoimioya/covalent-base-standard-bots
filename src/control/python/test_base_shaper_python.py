#!/usr/bin/env python3
"""
Test script for BaseShaper C++ implementation via Python bindings.
This script tests the same functionality as the C++ test_base_shaper.cpp.
"""

import numpy as np
import sys
import os

# Add the build directory to the Python path to find the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp', 'control'))

try:
    import build.src.cpp.control.base_shaper as base_shaper
    print("Successfully imported base_shaper module")
except ImportError as e:
    print(f"Failed to import base_shaper module: {e}")
    print("Make sure the C++ module has been compiled with Python bindings")
    sys.exit(1)

def test_basic_functionality():
    """Test basic BaseShaper functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Create BaseShaper with 0.001s sampling time
        shaper = base_shaper.BaseShaper(0.001)
        print("✓ BaseShaper created successfully")
        
        # Test single mode parameters (natural frequency = 10 rad/s, damping = 0.1)
        params = np.array([[10.0, 0.1]])
        
        # Test single sample shaping
        input_sample = 1.0
        shaped_sample = shaper.shape_sample(input_sample, params)
        
        print(f"✓ Input sample: {input_sample}")
        print(f"✓ Shaped sample: {shaped_sample}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_trajectory_shaping():
    """Test trajectory shaping functionality"""
    print("\n=== Testing Trajectory Shaping ===")
    
    try:
        # Create BaseShaper
        shaper = base_shaper.BaseShaper(0.001)
        
        # Test parameters
        params = np.array([[10.0, 0.1]])
        
        # Create a simple step trajectory
        num_samples = 100
        trajectory = np.zeros(num_samples)
        trajectory[50:] = 1.0  # Step at sample 50
        
        # Create varying parameters (same for all samples in this test)
        varying_params = [params for _ in range(num_samples)]
        
        # Shape the trajectory
        shaped_trajectory = shaper.shape_trajectory(trajectory, varying_params)
        
        print(f"✓ Original trajectory size: {trajectory.size}")
        print(f"✓ Shaped trajectory size: {shaped_trajectory.size}")
        print(f"✓ Trajectory shaping completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory shaping test failed: {e}")
        return False

def test_zvd_shaper_computation():
    """Test ZVD shaper computation"""
    print("\n=== Testing ZVD Shaper Computation ===")
    
    try:
        # Create BaseShaper
        shaper = base_shaper.BaseShaper(0.001)
        
        # Test parameters
        params = np.array([[10.0, 0.1]])
        
        # Compute ZVD shaper
        impulse_response, filter_order = shaper.compute_zvd_shaper(params)
        
        print(f"✓ ZVD shaper computed successfully!")
        print(f"✓ Impulse response length: {len(impulse_response)}")
        print(f"✓ Filter order: {filter_order}")
        
        return True
        
    except Exception as e:
        print(f"✗ ZVD shaper computation test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Create BaseShaper
        shaper = base_shaper.BaseShaper(0.001)
        
        # Test invalid sampling time
        try:
            invalid_shaper = base_shaper.BaseShaper(-0.001)
            print("✗ Should have thrown error for negative sampling time")
            return False
        except Exception as e:
            print(f"✓ Correctly caught error for negative sampling time: {e}")
        
        # Test invalid parameters (empty array)
        try:
            empty_params = np.array([]).reshape(0, 2)
            shaper.shape_sample(1.0, empty_params)
            print("✗ Should have thrown error for empty parameters")
            return False
        except Exception as e:
            print(f"✓ Correctly caught error for empty parameters: {e}")
        
        # Test invalid parameters (wrong shape)
        try:
            wrong_shape_params = np.array([[1.0, 2.0, 3.0]])  # 3 columns instead of 2
            shaper.shape_sample(1.0, wrong_shape_params)
            print("✗ Should have thrown error for wrong parameter shape")
            return False
        except Exception as e:
            print(f"✓ Correctly caught error for wrong parameter shape: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_multi_mode_shaper():
    """Test multi-mode shaper computation"""
    print("\n=== Testing Multi-Mode Shaper ===")
    
    try:
        # Create BaseShaper
        shaper = base_shaper.BaseShaper(0.001)
        
        # Test multi-mode parameters
        multi_params = np.array([
            [10.0, 0.1],  # First mode
            [20.0, 0.05]  # Second mode
        ])
        
        # Compute multi-mode ZVD shaper
        impulse_response, filter_order = shaper.compute_zvd_shaper(multi_params)
        
        print(f"✓ Multi-mode ZVD shaper computed successfully!")
        print(f"✓ Impulse response length: {len(impulse_response)}")
        print(f"✓ Filter order: {filter_order}")
        
        # Test shaping with multi-mode parameters
        input_sample = 1.0
        shaped_sample = shaper.shape_sample(input_sample, multi_params)
        print(f"✓ Multi-mode shaping completed: {shaped_sample}")
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-mode shaper test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing BaseShaper C++ implementation via Python bindings...")
    
    tests = [
        test_basic_functionality,
        test_trajectory_shaping,
        test_zvd_shaper_computation,
        test_error_handling,
        test_multi_mode_shaper
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
