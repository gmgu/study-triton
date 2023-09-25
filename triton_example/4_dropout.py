import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    # Dropout: for each element in an array, set the element to zero in probability p
    # x_ptr: input array
    # x_keep_ptr: x_keep_ptr[i] = 1 if keep, 0 if set to zero.
    # n_elements: len(x_ptr)
    # p: probability
    # BLOCK_SIZE: size of block that a kernel handles

    pid = tl.program_id(axis=0)  # get block id
    block_start = pid * BLOCK_SIZE  # 
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # we scale the output 1 / (1 - p) for consistent norm size during training and inference.
    output = tl.where(x_keep, x / (1 - p), 0.0)  # tl.where(condition, x, y): returns a tensor from x or y depending on condition
                                                 # shape of x and y are both broadcast to the shape of condition

    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, )).cuda()
p = 0.5
x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
output = dropout(x, x_keep, p)
print(tabulate.tabulate([["input"] + x.tolist(),
                         ["keep mask"] + x_keep.tolist(),
                         ["output"] + output.tolist()]))
