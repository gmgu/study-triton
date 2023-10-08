import torch
import triton
import triton.language as tl

@triton.jit
def rand_init(out_ptr, n, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    random = tl.rand(seed, offsets)
    tl.store(out_ptr + offsets, random, mask=mask)


out = torch.zeros(size=(10,)).to('cuda')
print('initial out', out)
n= out.numel()

grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
rand_init[grid](out, n, seed=999, BLOCK_SIZE=2)
print('out with seed=999', out)

rand_init[grid](out, n, seed=999, BLOCK_SIZE=2)
print('out with seed=999', out)

rand_init[grid](out, n, seed=777, BLOCK_SIZE=2)
print('out with seed=777', out)
