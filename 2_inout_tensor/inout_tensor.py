import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(in_ptr, out_ptr, n: tl.constexpr):
    offsets = tl.arange(0, n)   # 0, 1, 2, ... 7  (here, n must be power of 2 for tl.range)
    x = tl.load(in_ptr + offsets)  # load data (that pointer arrays are pointing) to x
    y = tl.store(out_ptr + offsets, x)  # save x to memory where pointer arrays are pointing


in_tensor = torch.rand(8, device='cuda')
out_tensor = torch.empty_like(in_tensor)
print('Input: ', in_tensor)
print('Output before call: ', out_tensor)

grid = lambda meta: (1, )
copy_kernel[grid](in_tensor, out_tensor, len(in_tensor))
print('Output after call: ', out_tensor)
