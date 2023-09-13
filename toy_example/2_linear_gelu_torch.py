import torch
import torch.nn as nn
import time


device = torch.device('cuda:0')

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(10, 1)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)

        return x


model = LinearModel().to(device)

data_x = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)

start_time = time.time()
for _ in range(1_000_000):
    pred_y = model(data_x)
end_time = time.time()

print(pred_y)
print(f'Elapsed Time: {end_time - start_time:.2f} sec')
