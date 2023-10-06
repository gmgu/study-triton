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
    # each thread computes out[bs0...bs0 + BLOCK_SIZE][bs1...bs1 + BLOCK_SIZE]
    #                      = x[bs0...bs0 + BLOCK_SIZE][:] * y[:][bs1...bs1 + BLOCK_SIZE]
    # x[range][:] * y[:][range] is again computed by x[range][0...BLOCK_SIZE] * y[0...BLOCK_SIZE][range] + ...
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    x_row_start = tl.arange(0, BLOCK_SIZE)[:, None] * m  # (b x 1), m is stride for x
    x_col_start = tl.arange(0, BLOCK_SIZE)[None, :]  # (1 x b)
    x_col_unit = BLOCK_SIZE  # we will add x_col_unit to x_col_start (shift from left to right)

    y_row_start = tl.arange(0, BLOCK_SIZE)[:, None] * p  # (b x 1), p is stride for y
    y_col_start = tl.arange(0, BLOCK_SIZE)[None, :]  # (1 x b)
    y_row_unit = BLOCK_SIZE * p  # we will add y_row_unit to y_row_start (shift from up to down)

    o_row_start = pid0 * BLOCK_SIZE * p + tl.arange(0, BLOCK_SIZE)[:, None] * p  # (b x 1)
    o_col_start = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]  # (1 x b)
    o_offset = o_row_start + o_col_start  # (b x b)

    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(tl.cdiv(m, BLOCK_SIZE)):
        # compute x_block and y_block with updated x_col and y_row
        x_block = x_row_start + x_col_start  # (b x b) broadcasted
        y_block = y_row_start + y_col_start  # (b x b) broadcasted

        x_offset = pid0 * BLOCK_SIZE * m + x_block
        x_mask = x_row_start < n * m and x_col_start < m  # n * m considers stride m

        y_offset = pid1 * BLOCK_SIZE + y_block
        y_mask = y_row_start < m * p and y_col_start < p  # m * p considers stride p

        x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
        y = tl.load(y_ptr + y_offset, mask=y_mask, other=0.0)
        out += tl.dot(x, y, allow_tf32=False)

        x_col_start += x_col_unit  # left to right
        y_row_start += y_row_unit  # up to down
    

    o_mask = o_row_start < n * p and o_col_start < p  # n * p considers stride p
    tl.store(out_ptr + o_offset, out, mask=o_mask)


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
        ms = tt.do_bench(lambda: triton_mm[grid](x, y, out, n, m, p, BLOCK_SIZE=64))

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
triton_mm[grid](x, y, out_triton, n, m, p, BLOCK_SIZE=64)

#print(out_torch)
#print(out_triton)
print(torch.allclose(out_torch, out_triton))

# performance measure
compare.run(show_plots=True, print_data=True)
