import tabulate
import torch

import triton
import triton.language as tl


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    # in the seeded dropout, we compute x_keep using seed and tl.rand
    # then, we can safely get x_keep during backpropagation, 
    # and easy to get the same results with the same seed
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # compute x_keep
    random = tl.rand(seed, offsets)  # rand(seed, offset): return uniform random float32 in (0, 1) for offset
    x_keep = random > p

    # scaling
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, )).cuda()

output1 = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=999)


print(tabulate.tabulate([["input"] + x.tolist(),
                         ["output (seed=123)"] + output1.tolist(),
                         ["output (seed=123)"] + output2.tolist(),
                         ["output (seed=999)"] + output3.tolist()]))
