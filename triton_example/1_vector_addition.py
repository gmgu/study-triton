import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)

  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)  # tl.arange(start, end): return contiguous values [start, ..., end)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)  # tl.load(pointer, mask, ...): return a tensor of data whose values are loaded from memory at location defined by pointer
                                           # pointer could be a single element pointer, then scalar will be loaded
                                           # pointer could be element-wise tensor of pointers
                                           # pointer could be a block pointer defined by make_block_ptr
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y

  tl.store(output_ptr + offsets, output, mask=mask)  # tl.store(pointer, value, mask, ...): store a tensor of data into memory locations defined by pointer


def add(x: torch.Tensor, y: torch.Tensor):
  output = torch.empty_like(x)
  assert x.is_cuda and y.is_cuda and output.is_cuda
  n_elements = output.numel()

  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

  return output

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['size'],  # x-axis label
    x_vals=[2**i for i in range(12, 28, 1)],
    x_log=True,
    line_arg='provider',
    line_vals=['triton', 'torch'],
    line_names=['Triton', 'Torch'],
    styles=[('blue', '-'), ('green', '-')],
    ylabel='GB/s',  # y-axis label
    plot_name='vector-add-performance',
    args={},
  )
)
def benchmark(size, provider):
  x = torch.rand(size, device='cuda', dtype=torch.float32)
  y = torch.rand(size, device='cuda', dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
  gbps = lambda ms: 12 * size / ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)


torch.manual_seed(0)
size = 999
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print('Diff: ', torch.max(torch.abs(output_torch - output_triton)).item())

benchmark.run(print_data=True, show_plots=True)