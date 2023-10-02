## How to input/output tensors to/from kernel
In this lecture, we learn how to give an input tensor to a kernel and get an output tensor from a kernel.
Similar to the CUDA kernel, Triton kernel gets input and returns output via argument.
One major difference is that arguments of triton.jit'd function are implicitly converted to pointers if they have a .data_ptr() method and a .dtype attribute (Torch.Tensor has both .data_ptr() and .dtype). data_ptr() returns the address of the first element of the tensor. dtype is the data type of the tensor (e.g., torch.float32). 


## Example: copy input tensor to output tensor
The following script copies in_tensor to out_tensor. Note that all threads are accessing the same data point, so there is no performance gain of using multiple threads. We will learn how to access different data point for each thread later.

```bash
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(in_ptr, out_ptr, n: tl.constexpr):
    offsets = tl.arange(0, n)   # 0, 1, 2, ... 7  (here, n must be power of 2 for tl.arange)
    x = tl.load(in_ptr + offsets)  # load data (that pointer arrays are pointing) to x
    y = tl.store(out_ptr + offsets, x)  # save x to memory where pointer arrays are pointing


in_tensor = torch.rand(8, device='cuda')
out_tensor = torch.empty_like(in_tensor)
print('Input: ', in_tensor)
print('Output before call: ', out_tensor)

grid = lambda meta: (1, )
copy_kernel[grid](in_tensor, out_tensor, len(in_tensor))
print('Output after call: ', out_tensor)
```

Regarding copy_kernel(), in_ptr (corr. out_ptr) is the address of the first element of in_tensor (corr. out_tensor). tl.load(addresses) returns the array of values that are stored in the addresses. Note that we need all addresses of the data, not the first address and the length. tl.store(addresses, values) stores the values to the addresses.


## Let's do the same job in parallel
In the following script, we have 8 blocks, where each block is an area that a thread should process, and each block corresponds to one array position. We can get the program id (thread id) by triton.language.program_id() function, in which we can specify the axis of the 3D launch grid. Since we are dealing with 1D array, axis=0. Now, each thread access distinct data point using the program id. Also, special efforts needed to be done to avoid out of index problem (see mask).

```bash
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # here, BLOCK is not a group of thread
    # BLOCK is an area that a thread should process
    pid = tl.program_id(axis=0)  # returns the id of current program instance
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n  # pid may be bigger than 7. prevent out of index.
    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.store(out_ptr + offsets, x, mask=mask)


in_tensor = torch.rand(8, device='cuda')
out_tensor = torch.empty_like(in_tensor)
print('Input: ', in_tensor)
print('Output before call: ', out_tensor)

grid = lambda meta: (8, )  # there are 8 blocks need to be processed, where 1 block is 1 position.
copy_kernel[grid](in_tensor, out_tensor, len(in_tensor), BLOCK_SIZE=1)
print('Output after call: ', out_tensor)
```
