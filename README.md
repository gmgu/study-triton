# study-trident
This repository is for studying Kakao Brain's Trident, which is an efficient library that can replace some PyTorch components. 


## Contents

- 1_hello_triton: a kernel that prints "Hello Triton Kernel!". explains func\[grid\]() syntax.
- 2_inout_tensor: how to input/output tensors to/from a kernel
- 3_measuring_performance: measure the performances of copy methods in PyTorch in terms of elapsed time using triton.testing library.
- 4_2d_grid_and_matmul: a kernel that computes matrix multiplication. 
- 5_out_of_index: how to handle out of index problem with an example of matrix multiplication.
- 6_random: using random seed in triton.jit'd function with an example of random tensor initialization.

## To be writen
- 7_pybind11: how to use C++ code in Python
