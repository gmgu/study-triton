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
