## Compare performance of different implementations
In the lecture, we learn how to compare performances between different implementations in terms of elapsed time.
Often times we need to compare the performance of our implementation using Triton and that using pure PyTorch (or parallelism vs sequential).
Triton provides the exact library for performance comparision.


## Example: performances of PyTorch copy methods
In thie example, we will compare copy methods provided by PyTorch. Let x and y be PyTorch tensors. We want to copy x to y.
There are mainly three ways to do this.
- y = x.clone().detach()
- y = torch.empty_like(x).copy\_(x)
- y = torch.tensor(x)

```bash
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
```

## @triton.testing.perf_report
@triton.testing.perf_report defines a Mark class with given function and triton.testing.Benchmark object, where Benchmark specifies benchmarking configuration.

```bash
def perf_report(benchmarks):
    return lambda fn: Mark(fn, benchmarks)
```

## triton.testing.Mark
Mark is a class for executing fn, collecting results for the executions, and printing results using matplotlib.pyplot and pandas.
We can start the performance measuring by calling Mark.run() method.
Note that we can also save results to plot_name.png (for image), results.html (showing the image), and plot_name.csv (for data) by setting Mark.run(save_path).


## triton.testing.Benchmark
Benchmark specifies inputs of the function by x_names and x_vals. Since input parameter can be more than one, x_names is a list to string. x_vals can be a list or list of lists (for the case when len(x_names) > 1). We can speficy different methods by line_arg. Note that line_arg is a single string and not a list of strings. Mark.run will loop through line_vals and x_vals. The rest of the parameters are for ploting.


## triton.testing.do_bench
triton.testing.do_bench(fn, quantiles, return_mode, ...) executes fn multiple times (including warmups that are not counted in the results). If quntiles=None, do_bench returns the mean elapsed time. If quantiles is not None, it should be an ordered triple and represents (mean, min, max) because Mark.run accepts only scalar or triple with that meaning. Note that getattr(torch, "mean")(times) is exactly the same as torch.mean(times). do_bench can be summarized as follows.

```bash
def do_bench(fn, quantiles=None, return_mode="mean", ...):
    for _ in range(n_warmup):
        fn()
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    for i in range(n_repeat):
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            rest = ret[0]
        return ret

    return getattr(torch. return_mode)(times).item()  # getattr(object, 'name') returns the member of object for the name
```
