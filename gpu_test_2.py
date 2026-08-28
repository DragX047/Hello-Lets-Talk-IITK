import torch
import time

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

N = 8192

a = torch.randn(N, N, device=device, dtype=torch.float16)
b = torch.randn(N, N, device=device, dtype=torch.float16)

# Warmup
for _ in range(20):
    c = a @ b

torch.cuda.synchronize()

iterations = 100

start = time.perf_counter()

for _ in range(iterations):
    c = a @ b

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

flops = 2 * N**3 * iterations
tflops = flops / elapsed / 1e12

print()
print("=" * 50)
print("FP16 TENSOR CORE TEST")
print("=" * 50)
print(f"Total time       : {elapsed:.2f} s")
print(f"Time / iteration : {elapsed / iterations * 1000:.2f} ms")
print(f"Approx FP16      : {tflops:.2f} TFLOPS")
print(
    f"Peak VRAM        : "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)