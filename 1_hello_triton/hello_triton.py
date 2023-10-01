import torch
import triton


@triton.jit
def hello_kernel():
    print("Hello Triton Kernel!")


torch.rand(1, device='cuda')  # for valid cuda resource handle
grid = lambda meta: (1, )  # number of blocks
hello_kernel[grid](num_warps=1)  # 1 warp has 32 threads
