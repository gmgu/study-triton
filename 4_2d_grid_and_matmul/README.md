## 2D grid and maxtrix multiplication
In this lecture, we learn how to set 2D grid with an example script for matrix multiplication.
2D grid is useful when we deal with 2D tensor such as matrices.

## Example: matrix multiplication

In the following script, there are two functions that compute the matrix multiplication.
torch_mm calls torch.mm, which is a PyTorch implementation of matrix multiplication.
triton_mm is a naive implementation of matrix multiplication using Triton. The implementation is naive because elements of a row of x (and a column of y) is not parallelized (we only parallelized rows and columns, not the elements of them).
Inside of triton_mm, we use 2D grid (pid0, pid1) to identify which (BLOCK_SIZE x BLOCK_SIZE) block of out should be computed by a thread.

```bash
import torch
import triton
import warnings
import triton.testing as tt
import triton.language as tl
warnings.filterwarnings("ignore")


def torch_mm(x, y):
    return torch.mm(x, y)


@triton.jit
def triton_mm(x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # x: n x m, y: m x p, out: n x p
    # a naive implementation: each thread computes out[bs0...bs0 + BLOCK_SIZE][bs1...bs1 + BLOCK_SIZE]
    #                         = x[bs0...bs0 + BLOCK_SIZE][:] * y[:][bs1...bs1 + BLOCK_SIZE]
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    x_row = pid0 * BLOCK_SIZE * m + tl.arange(0, m)
    x_col = tl.arange(0, BLOCK_SIZE) * m
    x_offset = x_row[None, :] + x_col[:, None]  # (1 x m) + (b x 1) = (b x m)
    x_mask = tl.core.full((1, m), True, dtype=tl.int1) and \
             (pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < n

    y_row = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y_col = tl.arange(0, m) * p
    y_offset = y_row[None, :] + y_col[:, None]  # (1 x b) + (m x 1) = (m x b)
    y_mask = (pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[None, :] < p and \
             tl.core.full((m, 1), True, dtype=tl.int1)

    x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
    y = tl.load(y_ptr + y_offset, mask=y_mask, other=0.0)
    out = tl.dot(x, y, allow_tf32=False)

    out_row = pid0 * BLOCK_SIZE * p + pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_col = tl.arange(0, BLOCK_SIZE) * p
    out_offset = out_row[None, :] + out_col[:, None]  # (1 x b) + (b x 1) broadcasted to (b x b)
    out_mask = (pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[None, :] < p and \
               (pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < n

    tl.store(out_ptr + out_offset, out, mask=out_mask)
            

@tt.perf_report(
    tt.Benchmark(x_names=['p'],
              x_vals=[2**i for i in range(5, 20)],
              line_arg='method',
              line_vals=['torch_mm', 'triton_mm'],
              line_names=['torch.mm(x, y)', 'naive Triton mm implementation'],
              plot_name='Torch.mm vs naive Triton mm implementation',
              xlabel='p',
              ylabel='Avg. Elapsed Time (ms)',
              x_log=True,
              y_log=False,
              args={},
    )
)
def compare(p, method):
    n = 32
    m = 32
    x = torch.rand((n, m), device='cuda', dtype=torch.float32)
    y = torch.rand((m, p), device='cuda', dtype=torch.float32)
    out = torch.zeros((n, p), device='cuda', dtype=torch.float32)

    if method == 'torch_mm':
        ms = tt.do_bench(lambda: torch.mm(x, y))
    elif method == 'triton_mm':
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), triton.cdiv(p, meta['BLOCK_SIZE']))
        ms = tt.do_bench(lambda: triton_mm[grid](x, y, out, n, m, p, BLOCK_SIZE=32))

    return ms


## main
torch.manual_seed(9)

# check if two methods output the same value
n = 16
m = 16
p = 32
x = torch.rand((n, m), device='cuda', dtype=torch.float32)
y = torch.rand((m, p), device='cuda', dtype=torch.float32)

out_torch = torch_mm(x, y)

grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), triton.cdiv(p, meta['BLOCK_SIZE']))
out_triton = torch.rand((n, p), device='cuda', dtype=torch.float32)
triton_mm[grid](x, y, out_triton, n, m, p, BLOCK_SIZE=32)

print(torch.allclose(out_torch, out_triton))

# performance measure
compare.run(show_plots=True, print_data=True)
```

## class triton.language.constexpr
constexpr is a class in Triton that is used to store a value that is known at compile-time. Some functions in Triton gets input of type constexpr.


## triton.language.core.full()
triton.language.core.full is a builtin function that returns a tensor filled with the scalar value for the given shape and dtype. The tensor class in Triton is different to the tensor in PyTorch, and they are not compatible. That is you cannot add a PyTorch tensor to a Triton tensor. Each dimention of shape must be int or constexpr[int]. In semantic.full(), only size-1 tensor or scalar is accepted for value, and dtype must be specified when value is not a tensor.
```bash
def full(shape, value, dtype, _builder):
    shape = _shape_check_impl(shape)  # assert all dimentions are in type constexpr[int] or int and return an integer array.
    value = _constexpr_to_value(value)  # returns constexpr.value (i.e., value.value) if isinstance(value, constexpr). otherwise return identity (i.e., value).
    dtype = _constexpr_to_value(dtype)
    return semantic.full(shape, value, dtype, _builder)
```


## triton.language.core.dot()
triton.language.core.dot() returns the matrix product of two blocks.
The two blocks must be two dimentional, shape compatible (i.e., lhs.shape[1].value == rhs.shape[0].value), all values in both first input shape and second input shape must be >=16, 

tf32 is tensor float 32 format for float point representaion. The general fp32 format uses 1 bit for sign, 8 bits for range, and 23 bits for precision. fp16 format uses 1 bit for sign, 5 bits for range, and 10 bits for precision. Google Brain's bfloat16 format uses 1 bit for sign, 8 bits for range (same as fp32), and 7 bits for precision. tf32 format uses 1 bit for sign, 8 bits for range (same as fp32), and 10 bits for precision (same as fp16). tf32 uses 19 bits in total (so it is fast), has the same range as fp32 and bf16 (so it does not need scaling), has the same precision as fp16 (so it is accurate enough).

In default, allow_tf32 is True in the triton.language.core.dot(). In PyTorch, allow_tf32 flag is True in default from PyTorch 1.7 to PyTorch 1.11, and false in default from PyTorch 1.12. Therefore, the results of dot product computed by Triton may be differnt with that computd by PyTorch depending on the versions.
