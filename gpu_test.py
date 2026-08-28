import torch
import time

print("=" * 50)
print("PyTorch GPU Stress Test")
print("=" * 50)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"CUDA version    : {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\nCUDA is NOT available!")
    exit()

device = torch.device("cuda")
gpu = torch.cuda.get_device_name(0)

props = torch.cuda.get_device_properties(0)

print(f"GPU              : {gpu}")
print(f"VRAM             : {props.total_memory / 1024**3:.2f} GB")
print()

# Matrix size
N = 8192

print(f"Creating {N} x {N} matrices...")

try:
    a = torch.randn((N, N), device=device, dtype=torch.float32)
    b = torch.randn((N, N), device=device, dtype=torch.float32)

    print(
        f"VRAM allocated   : "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print("\nWarming up GPU...")

    for _ in range(10):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()

    print("Running benchmark...\n")

    iterations = 50

    start = time.perf_counter()

    for i in range(iterations):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)

    print(f"Iterations       : {iterations}")
    print(f"Total time       : {elapsed:.2f} seconds")
    print(f"Time / iteration : {elapsed / iterations * 1000:.2f} ms")
    print(f"GPU              : {gpu}")

    print(
        f"VRAM allocated   : "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        f"Peak VRAM        : "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

    # Rough FP32 throughput
    flops_per_matmul = 2 * (N ** 3)
    total_flops = flops_per_matmul * iterations
    tflops = total_flops / elapsed / 1e12

    print(f"Approx FP32      : {tflops:.2f} TFLOPS")

    print("=" * 50)

except torch.cuda.OutOfMemoryError:
    print("\nGPU ran out of VRAM.")
    print("That's not a failure — we'll reduce the matrix size.")

finally:
    if "a" in locals():
        del a
    if "b" in locals():
        del b
    if "c" in locals():
        del c

    torch.cuda.empty_cache()