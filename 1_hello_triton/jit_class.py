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
