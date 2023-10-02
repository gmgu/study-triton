## What is Triton
Triton is a python library for GPU programming.
It is similar to the CUDA C++ language. In this lecture, we will learn how to code Triton program.

## A Triton kernel that prints "Hello Triton Kernel!"
Let's write a simple python script that prints "Hello Triton Kernel!" using Triton.

```bash
import torch
import triton


@triton.jit
def hello_kernel():
    print("Hello Triton Kernel!")


torch.rand(1, device='cuda')  # for valid cuda resource handle
grid = lambda meta: (1, )  # how many blocks do we need to process
hello_kernel[grid](num_warps=1)  # 1 warp has 32 threads
```

If you run this code, you will see 32 prints of "Hello Triton Kernel!".
We note that torch.rand is required only for the cuda recource handle.

## CUDA C++ vs Triton
In CUDA C++, a kernel is executed n times in parallel by n threads. A grid of the kernel consists of NB blocks, where each block consists of NT threads and NB * NT = n.

In Triton, a grid of a kernel consists of NB blocks. Each thread process one block (so, the meaning of 'block' in Triton is different from that in CUDA). GPU can compute multiple threads ,num_warps * 32 threads, in parallel.

In the "Hello Triton Kernel!" example, we set the number of blocks to 1 and num_warps to 1, and thus the kernel is executed 32 times by 32 threads.


## What is @triton.jit (and function\[grid\]() syntax)
@triton.jit decorator is a template class. Here is a part of the code for triton.jit.

```bash
class KernelInterface(Generic[T]):
  run: T
  def __getitem__(self, grid):
    return cast(T, functools.partial(cast(Callable, self.run), grid=grid))
...

class JITFunction(KernelInterface[T]):
  def __init__(...):
    self.run = self._make_launcher()

  def _make_launcher(...):
    def launcher_body(args_proxy, grid, num_warps, ...):
      ...
    exec(src, scope)
    return scope[self.gn.__name__]
...

def jit(fn, ...):
  def decorator(fn: T) -> JITFunction[T]:
    ...
  return decorator(fn)
...
```

Essentially, @triton.jit defines a class JitFunction for given function fn with the function type T.
For example, by stating hello_kernel[grid], we create an instance of the JitFunction class with fn = hello_kernel and T = grid.
This will return JitFunction.run function due to \__getitem__ function. Then, hello_kernel\[grid\](args) is equal to JitFunction.run(grid=grid, args). In the _make_launcher() function you can check possible arguments for the JitFunction.run() function, including GPU specific arguments like num_warps.

## Example of @triton.jit
Here is another version of @triton.jit decorator for demonstration.
We defined @python_jit decorator that defines JitFunction class. By calling print_hello\[grid\](), we instantiate JitFunction class (due to print_hello\[grid\]), get JitFunction.run function (due to overloading \__getitem__), and call JitFunction.run function (due to round bracket). Then, JitFunction.run will call the given function of print_hello.

```bash
import functools
from typing import Generic, TypeVar, Callable, cast

T = TypeVar('T')

class KernelInterface(Generic[T]):
    run: T
    def __getitem__(self, grid):
        return cast(T, functools.partial(cast(Callable, self.run), grid=grid))

class JitFunction(KernelInterface[T]):
    def __init__(self, fn):
        self.fn = fn

    def run(self, grid):
        self.fn()

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")


def python_jit(fn):
    return JitFunction(fn)


@python_jit
def print_hello():
    print("hello world!")


grid = lambda meta: (1, )
print_hello[grid]()
```


## Calling Triton kernel in another way.

We can seperate the kernel call by two stages.
1. get function by fn = hello_kernel\[grid\]
2. call the function by fn(num_warps=1)

```bash
import torch
import triton


@triton.jit
def hello_kernel():
    print("Hello Triton Kernel!")


torch.rand(1, device='cuda')  # for valid cuda resource handle

# we can also call the function like this!
grid = lambda meta: (1, )  # just like before
fn = hello_kernel[grid]  # hello_kernel[grid] returns a JitFunction.run function
fn(num_warps=1)  # and we call it!
```

Nevertheless, I think func\[grid\]() syntax is more appropriate to use because it has a correspondense with func<<<...>>>() syntax of CUDA C++.

