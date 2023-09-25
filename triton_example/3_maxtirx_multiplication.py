import torch

import triton
import triton.language as tl

# we will revisit autotune later
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
                  GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
  # C[M, N] = A[M, K] x B[K, N]

  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # cdiv(x, div) returns (x + div - 1) // div
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n  # GROUP_SIZE_M: the number of blocks in terms of M to be computed together
  group_id = pid // num_pid_in_group  # 'pid divided by the group size' is the group_id that pid belongs
  first_pid_m = group_id * GROUP_SIZE_M  # the first pid of group_id
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # GROUP_SIZE_M may not divide num_pid_m, 
                                                             # so the actual group size of group_id may smaller than GROUP_SIZE_M
  pid_m = first_pid_m + (pid % group_size_m)  # what if group_size_m < GROUP_SIZE_M, and thus there exists same pid_m?
  pid_n = (pid % num_pid_in_group) // group_size_m  # ?

  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, BLOCK_SIZE_K)
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)  # k * BLOCK_SIZE_K is current addr for k
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
    accumulator = accumulator + tl.dot(a, b)  # returns the matrix product of two blocks
    a_ptrs = a_ptrs + BLOCK_SIZE_K * stride_ak  # recall that X[i, j] = X + i * stride_m + j * stride_k
                                                # that is, X[i, j + k] = X + i * stride_m + (j + k) * stride_k
                                                #                      = X[i, j] + k * stride_k
    b_ptrs = b_ptrs + BLOCK_SIZE_K * stride_bk  # likewise, X[i + k, j] = X[i, j] + k * stride_m (stride_m for b is stride_bk)

  if ACTIVATION == 'leaky_relu':
    accumulator = leaky_relu(accumulator)
  c = accumulator.to(tl.float16)

  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
  # LeakyReLU(x) = x if x >= 0 else negative_slope * x
  x = x + 1  # why add 1? it will make LeakyReLU(x) = x + 1 if x >= -1 else negative_slope * (x + 1)
  return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
  assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
  assert a.is_contiguous(), "Matrix A must be contiguous"
  assert b.is_contiguous(), "Matrix B must be contiguous"
  M, K = a.shape
  K, N = b.shape
  c = torch.empty((M, N), device=a.device, dtype=a.dtype)
  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
  
  #print(a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
  matmul_kernel[grid](a, b, c, M, N, K, 
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                      ACTIVATION=activation)
  return c


torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
  print("Triton and Torch match")
else:
  print("Triton and Torch differ")



@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],
    x_vals=[128 * i for i in range(2, 33)],
    line_arg='provider',
    line_vals=['cublas', 'triton'],
    line_names=['cuBLAS', 'Triton'],
    styles=[('green', '-'), ('blue', '-')],
    ylabel='TFLOPS',
    plot_name='matmul-performance',
    args={},
  )
)
def benchmark(M, N, K, provider):
  a = torch.randn((M, K), device='cuda', dtype=torch.float16)
  b = torch.randn((K, N), device='cuda', dtype=torch.float16)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'cublas':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
  perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
  return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
