#!/usr/bin/env python3
"""
compare_numpy_files.py

Compares two .npy files and asserts they are identical (within tolerance for floats).

Usage:
    python compare_numpy_files.py file1.npy file2.npy
"""

import sys
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_numpy_files.py <file1.npy> <file2.npy>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    # Load both arrays
    arr1 = np.load(file1, allow_pickle=True)
    arr2 = np.load(file2, allow_pickle=True)

    # Compare shape
    if arr1.shape != arr2.shape:
        raise AssertionError(f"Shape mismatch: {arr1.shape} != {arr2.shape}")

    # Compare dtype
    if arr1.dtype != arr2.dtype:
        raise AssertionError(f"Dtype mismatch: {arr1.dtype} != {arr2.dtype}")
    
    print(np.min(arr1, axis=0))
    print(np.min(arr2, axis=0))

    # Compare values
    if np.issubdtype(arr1.dtype, np.floating):
        if not np.allclose(arr1, arr2, rtol=0, atol=0):
            diff = np.max(np.abs(arr1 - arr2))
            raise AssertionError(f"Arrays differ (max abs diff = {diff})")
    else:
        if not np.array_equal(arr1, arr2):
            raise AssertionError("Arrays differ (non-floating type)")

    print("✅ Files match perfectly!")

if __name__ == "__main__":
    main()
