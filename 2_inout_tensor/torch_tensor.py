import torch

x = torch.rand(9)
print("x:", x)
print("x.data_ptr():", x.data_ptr())
print("x.dtype:", x.dtype)

for i in range(len(x)):
    print(f"x[{i}].data_ptr(): {x[i].data_ptr()}")
