import torch
import warnings
import triton.testing as tt
warnings.filterwarnings("ignore")


def clone_detach(x):
    return x.clone().detach()  # recommended way by PyTorch


def empty_copy(x):
    return torch.empty_like(x).copy_(x)


def tensor(x):
    return torch.tensor(x)


@tt.perf_report(  # generate line plots
    tt.Benchmark(x_names=['n'],  # name of the arguments that should appear on the x axis of the plot
              x_vals=[2**i for i in range(10, 20)],  # list of values to use for arguments in x_names
              line_arg='method',  # argument name for which different values correspond to different lines in the plot
              line_vals=['clone_detach', 'empty_copy', 'tensor'],  # list of values to use for the arguments in line_arg
              line_names=['x.clone().detach()', 'torch.empty_like(x).copy_(x)', 'torch.tensor(x)'],  # label names for the different lines
              plot_name='Performance of Copy Methods in PyTorch',  # name of the plot
              xlabel='Size of Array',  # label for the x axis of the plot
              ylabel='Avg. Elapsed Time (ms)',  # label for the y axis of the plot
              x_log=True,  # whether the x axis should be log scale
              y_log=False, # whether the y axis should be log scale
              args={},  # dict of keyword arguments to remain fixed throughout the benchmark
    )  # end Benchmark
)  # end perf_report
def compare(n, method):
    x = torch.rand(n, device='cuda', dtype=torch.float32)
    if method == 'clone_detach':
        ms = tt.do_bench(lambda: clone_detach(x))  # do_bench(fn, ...) runs fn few times for warmup and then run fn multiple times for measuring time
    elif method == 'empty_copy':
        ms = tt.do_bench(lambda: empty_copy(x))    # by default, do_bencn(..., quantiles=None, return_mode="mean"), and returns torch.mean(times).item()
    elif method == 'tensor':
        ms = tt.do_bench(lambda: tensor(x))        # so, ms is the mean time of the multiple runs.
    return ms


torch.manual_seed(9)
compare.run(show_plots=True, print_data=True)  # compare.run always show plots and then print data. defaults are false.
