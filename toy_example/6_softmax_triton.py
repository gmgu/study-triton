# given array A of size n, 
# softmax(i, A) = e^A[i] / sum(e^A[j] for j in range(n))

import torch
import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
  # compute row-wise softmax of matrix X, where X.shape = M, N

  # read MN elements (X) and write M elements (x_max)
  x_max = x.max(dim=1)[0]  # values, indices = torch.max()

  # read MN + M elements (x, x_max) and write MN elements (z)
  z = x - x_max[:, None]  # slicing with None adds an axis (M) -> (M, 1)

  # read MN elements (x) and write MN elements (numer)
  numer = torch.exp(x)
  
  # read MN elements (numer) and write M elements (denom)
  denom = numer.sum(dim=1)

  # read MN + M elements (numer, deom) and write MN elements (ret)
  ret = numer / denom[:, None]

  # total read 5MN + 2M elements and write 3MN + 2M elements
  return ret


@triton.jit
def triton_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
  row_idx = tl.program_id(0)

  row_start_ptr = input_ptr + row_idx * input_row_stride  # input_row_stride = N

  col_offsets = tl.arange(0, BLOCK_SIZE)

  input_ptrs = row_start_ptr + col_offsets

  row = tl.load(input_ptrs, mask=(col_offsets < n_cols), other=-float('inf'))  # if mask[idx] is false, return other[idx]

  row_minus_max = row - tl.max(row, axis=0)

  numer = tl.exp(row_minus_max)
  denom = tl.sum(numer, axis=0)
  softmax_output = numer / denom

  output_row_start_ptr = output_ptr + row_idx * output_row_stride  # output_row_stride = N
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=(col_offsets < n_cols))


def triton_softmax(x):
  M, N = x.shape

  # we set BLOCK_SIZE bigger than N, so a row is not divided
  BLOCK_SIZE = triton.next_power_of_2(N)  # smallest power of two greater than N

  num_warps = 4
  if BLOCK_SIZE >= 2048:
    num_warps = 8
  if BLOCK_SIZE >= 4096:
    num_warps = 16

  y = torch.empty_like(x)

  # each of M programs will handle a row of length N
  triton_softmax_kernel[(M, )](y, x, x.stride(0), y.stride(0), N, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)

  return y


torch.manual_seed(0)
x = torch.randn(1999, 999, device='cuda')
y_triton = triton_softmax(x)
y_torch = torch.softmax(x, axis=1)

assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['N'],  # columns
    x_vals=[128 * i for i in range(2, 100)],
    line_arg='provider',
    line_vals=['triton', 'torch-native', 'torch-jit'],
    line_names=['Triton', 'Torch (native)', 'Torch (jit)'],
    styles=[('blue', '-'), ('green', '-'), ('green', '--')],
    ylabel='GB/s',
    plot_name='softmax performance',
    args={'M': 4096}
  )  
)
def benchmark(M, N, provider):
  x = torch.randn(M, N, device='cuda', dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'torch-native':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
  if provider == 'torch-jit':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
  gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True)
