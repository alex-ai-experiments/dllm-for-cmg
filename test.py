import torch
import time

# 1. Setup - Create two large matrices (100 million numbers each)
size = 10000 
cpu_tensor1 = torch.randn(size, size)
cpu_tensor2 = torch.randn(size, size)

print(f"Matrix size: {size}x{size}")

# --- CPU BENCHMARK ---
start_cpu = time.time()
cpu_result = torch.matmul(cpu_tensor1, cpu_tensor2)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print(f"CPU Time: {cpu_time:.4f} seconds")

# --- GPU BENCHMARK ---
if torch.cuda.is_available():
    # Move tensors to GPU (your GTX 1650)
    gpu_tensor1 = cpu_tensor1.to('cuda')
    gpu_tensor2 = cpu_tensor2.to('cuda')
    
    # Warm-up (GPUs are slow on the very first call as they initialize)
    _ = torch.matmul(gpu_tensor1, gpu_tensor2)
    torch.cuda.synchronize() # Wait for GPU to finish warm-up

    start_gpu = time.time()
    gpu_result = torch.matmul(gpu_tensor1, gpu_tensor2)
    torch.cuda.synchronize() # Ensure we measure the full math operation
    end_gpu = time.time()
    
    gpu_time = end_gpu - start_gpu
    print(f"GPU Time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x faster on GPU!")
else:
    print("GPU not found. Skipping GPU benchmark.")