import torch
import time

device = torch.device('cuda:0')

def custom_mm(A, B):
  ra, ca = A.shape
  rb, cb = B.shape
  C = torch.zeros((ra, cb)).to(A.device)
  for i in range(ra):
    for j in range(cb):
      for k in range(rb):
        #C[i][j] += (A[i][k].item() * B[k][j].item())
        C[i][j] += (A[i][k] * B[k][j])

  return C

def torch_mm(A, B):
  return torch.mm(A, B)


ROW=32
COL=32
ITER=1

total_time = 0
for _ in range(ITER):
  A = torch.randn(ROW * COL).view(ROW, COL).to(device).to(torch.float32)
  B = torch.randn(ROW * COL).view(COL, ROW).to(device).to(torch.float32)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()

  C = torch_mm(A, B)

  end.record()
  torch.cuda.synchronize()

  total_time += (start.elapsed_time(end))
print(f'Torch.mm Elapsed Time: {total_time / 1000:.2f} sec')


total_time = 0
for _ in range(ITER):
  A = torch.randn(ROW * COL).view(ROW, COL).to(device).to(torch.float32)
  B = torch.randn(ROW * COL).view(COL, ROW).to(device).to(torch.float32)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()

  C = custom_mm(A, B)

  end.record()
  torch.cuda.synchronize()

  total_time += (start.elapsed_time(end))
print(f'Custom mm Elapsed Time: {total_time / 1000:.2f} sec')
