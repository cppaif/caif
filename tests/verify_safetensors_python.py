#!/home/aidev/pytorch_cuda/bin/python3
"""
Python Serialization Compatibility Test for AIF SafeTensors

This script verifies that SafeTensors files created by AIF can be read
by the Python safetensors library.

Run test_safetensors first to create the test file, then run this script.
"""

import sys
import numpy as np

def main():
    test_path = "/tmp/aif_python_compat.safetensors"

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors library not installed")
        print("Install with: pip install safetensors")
        return 1

    print("=== Python SafeTensors Compatibility Test ===\n")

    try:
        with safe_open(test_path, framework="numpy") as f:
            # Check metadata
            metadata = f.metadata()
            print(f"Metadata: {metadata}")

            if metadata is None:
                print("[FAIL] No metadata found")
                return 1

            if metadata.get("framework") != "AIF":
                print(f"[FAIL] Expected framework='AIF', got '{metadata.get('framework')}'")
                return 1

            print("[PASS] Metadata readable")

            # Check tensors
            tensor_names = f.keys()
            print(f"Tensors: {list(tensor_names)}")

            expected_tensors = {"model.weight", "model.bias"}
            if set(tensor_names) != expected_tensors:
                print(f"[FAIL] Expected tensors {expected_tensors}, got {set(tensor_names)}")
                return 1

            print("[PASS] Tensor names match")

            # Verify weight tensor
            weight = f.get_tensor("model.weight")
            expected_weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

            print(f"Weight shape: {weight.shape}, dtype: {weight.dtype}")
            print(f"Weight values:\n{weight}")

            if weight.shape != (2, 3):
                print(f"[FAIL] Weight shape mismatch: expected (2, 3), got {weight.shape}")
                return 1

            if not np.allclose(weight, expected_weight):
                print(f"[FAIL] Weight values mismatch")
                print(f"Expected:\n{expected_weight}")
                return 1

            print("[PASS] Weight tensor correct")

            # Verify bias tensor
            bias = f.get_tensor("model.bias")
            expected_bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

            print(f"Bias shape: {bias.shape}, dtype: {bias.dtype}")
            print(f"Bias values: {bias}")

            if bias.shape != (3,):
                print(f"[FAIL] Bias shape mismatch: expected (3,), got {bias.shape}")
                return 1

            if not np.allclose(bias, expected_bias):
                print(f"[FAIL] Bias values mismatch")
                print(f"Expected: {expected_bias}")
                return 1

            print("[PASS] Bias tensor correct")

    except FileNotFoundError:
        print(f"ERROR: Test file not found: {test_path}")
        print("Run ./test_safetensors first to create the test file")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("\n=== All Python compatibility tests passed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
