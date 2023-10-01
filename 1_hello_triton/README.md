## What is Triton
Triton is python library for GPU programming.
It is similar to the CUDA C++ language. In this lecture, we will learn how to code Triton program.

## A Triton kernel that prints "Hello Triton Kernel!"
Let's write a simple python script that print "Hello Triton Kernel!" using Triton.

```bash
import torch
import triton


@triton.jit
def hello_kernel():
    print("Hello Triton Kernel!")


torch.rand(1, device='cuda')  # for valid cuda resource handle
grid = lambda meta: (1, )  # number of blocks
hello_kernel[grid](num_warps=1)  # 1 warp has 32 threads
```

If you run this code, you will see 32 prints of "Hello Triton Kernel!".

## CUDA C++ vs Triton
In CUDA C++, a kernel is executed n times in parallel by n threads. A grid of the kernel consists of NB blocks, where each block consists of NT threads and NB * NT = n.

Similarly, in Triton, a grid of a kernel consists of NB blocks (1 in the example), where each block consists of num_warps * 32 threads.

In the "Hello Triton Kernel!" example, we set the number of blocks to 1 and num_warps to 1. Since there are 32 threads in a warp, there are 32 threads in total in the grid. Therefore, the kernel is executed 32 times by 32 threads.


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

Essentially, @triton.jit defines a class JitFunction for hello_kernel function with the function type T = grid.
By hello_kernel[grid], you create an instance of the class with T = grid.
This will return JitFunction.run function due to __getitem__ function (i.e., hello_kernel[grid] invokes KernelInterface.__getitem__(grid)). Then, hello_kernel\[grid\](args) is equal to JitFunction.run(grid=grid, args). In the _make_launcher() function you can check the arguments possible for the JitFunction.run() function, including GPU specific arguments like num_warps.

## Example for @triton.jit
Here is another version of @triton.jit decorator for demonstration.
We defined @python_jit decorator that defines JitFunction class. By calling print_hello\[grid\](), we instantiate JitFunction class (due to print_hello\[grid\]), get JitFunction.run function (due to overloading __getitem__) which is print_hello, and call JitFunction.run function will execute given function (due to round brackets).

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


