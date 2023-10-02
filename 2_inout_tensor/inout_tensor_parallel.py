import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # here, BLOCK is not a group of thread
    # BLOCK is an area that a thread should process
    pid = tl.program_id(axis=0)  # returns the id of current program instance
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n  # pid may be bigger than 7. prevent out of index.
    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.store(out_ptr + offsets, x, mask=mask)


in_tensor = torch.rand(8, device='cuda')
out_tensor = torch.empty_like(in_tensor)
print('Input: ', in_tensor)
print('Output before call: ', out_tensor)

grid = lambda meta: (8, )  # there are 8 blocks needed to be processed, where 1 block is 1 position.
copy_kernel[grid](in_tensor, out_tensor, len(in_tensor), BLOCK_SIZE=1)
print('Output after call: ', out_tensor)
