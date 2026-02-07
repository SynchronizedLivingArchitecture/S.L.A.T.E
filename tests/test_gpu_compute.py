# Modified: 2026-02-07T02:05:00Z | Author: COPILOT | Change: GPU compute validation test for CI
"""
SLATE GPU Compute Validation Test
Validates CUDA availability, runs matmul benchmarks on all GPUs,
and tests multi-GPU parallel compute.
"""

import sys
import time


def test_cuda_available():
    """Test that CUDA is available and PyTorch can see GPUs."""
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("WARN: CUDA not available - skipping GPU tests")
        return False

    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    return True


def test_single_gpu_compute():
    """Run matmul benchmark on each GPU individually."""
    import torch

    if not torch.cuda.is_available():
        print("SKIP: No CUDA")
        return True

    results = []
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"\nGPU {i}: {name} (compute {cap[0]}.{cap[1]}, {mem_gb:.1f} GB)")

        # Warm up
        _ = torch.randn(1024, 1024, device=dev)
        torch.cuda.synchronize(dev)

        # Benchmark: 4096x4096 matrix multiply
        a = torch.randn(4096, 4096, device=dev)
        b = torch.randn(4096, 4096, device=dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - t0
        tflops = 2 * 4096**3 / elapsed / 1e12
        print(f"  Matmul 4096x4096: {elapsed*1000:.1f}ms ({tflops:.2f} TFLOPS)")

        # Tensor allocation test
        big = torch.zeros(2048, 2048, 32, device=dev)
        alloc_mb = big.element_size() * big.nelement() / 1e6
        print(f"  Allocated {alloc_mb:.0f} MB tensor successfully")

        peak_mb = torch.cuda.max_memory_allocated(i) / 1e6
        print(f"  Peak memory: {peak_mb:.0f} MB")

        results.append({
            "gpu": i,
            "name": name,
            "tflops": round(tflops, 2),
            "ms": round(elapsed * 1000, 1),
        })

        del a, b, c, big
        torch.cuda.empty_cache()

    print(f"\n=== ALL {len(results)} GPUs VALIDATED ===")
    for r in results:
        print(f"  GPU {r['gpu']}: {r['tflops']} TFLOPS ({r['ms']}ms)")

    return True


def test_multi_gpu_parallel():
    """Run matmul on both GPUs simultaneously."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIP: Need 2+ GPUs for multi-GPU test")
        return True

    print("=== Multi-GPU Parallel Compute ===")
    a0 = torch.randn(4096, 4096, device="cuda:0")
    b0 = torch.randn(4096, 4096, device="cuda:0")
    a1 = torch.randn(4096, 4096, device="cuda:1")
    b1 = torch.randn(4096, 4096, device="cuda:1")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    c0 = torch.matmul(a0, b0)
    c1 = torch.matmul(a1, b1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    combined_tflops = 2 * 2 * 4096**3 / elapsed / 1e12
    print(f"  Dual-GPU parallel matmul: {elapsed*1000:.1f}ms ({combined_tflops:.2f} combined TFLOPS)")
    print(f"  GPU 0 result shape: {c0.shape}, sum: {c0.sum().item():.2f}")
    print(f"  GPU 1 result shape: {c1.shape}, sum: {c1.sum().item():.2f}")

    del a0, b0, c0, a1, b1, c1
    torch.cuda.empty_cache()
    print("=== Multi-GPU PASSED ===")
    return True


def test_embedding_simulation():
    """Simulate transformer-style embedding computation on GPU."""
    import torch

    if not torch.cuda.is_available():
        print("SKIP: No CUDA")
        return True

    dev = torch.device("cuda:0")
    print("=== Embedding Generation (simulated) ===")
    batch_size = 64
    seq_len = 512
    hidden_dim = 768

    embeddings = torch.randn(batch_size, seq_len, hidden_dim, device=dev)
    weight = torch.randn(hidden_dim, hidden_dim, device=dev)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    projected = torch.matmul(embeddings, weight)
    projected = torch.nn.functional.normalize(projected, dim=-1)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    print(f"  {batch_size} sequences x {seq_len} tokens x {hidden_dim}d: {ms:.1f}ms")
    print(f"  Output shape: {projected.shape}")
    mem_mb = torch.cuda.memory_allocated(0) / 1e6
    print(f"  Memory used: {mem_mb:.0f} MB")

    del embeddings, weight, projected
    torch.cuda.empty_cache()
    print("=== Embedding Test PASSED ===")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  SLATE GPU Compute Validation")
    print("=" * 50)

    tests = [
        ("CUDA Availability", test_cuda_available),
        ("Single GPU Compute", test_single_gpu_compute),
        ("Multi-GPU Parallel", test_multi_gpu_parallel),
        ("Embedding Simulation", test_embedding_simulation),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\n--- {name} ---")
        try:
            result = func()
            if result:
                passed += 1
                print(f"  RESULT: PASS")
            else:
                failed += 1
                print(f"  RESULT: FAIL")
        except Exception as e:
            failed += 1
            print(f"  RESULT: ERROR - {e}")

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    sys.exit(1 if failed > 0 else 0)
