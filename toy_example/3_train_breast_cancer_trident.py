import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from torch.utils.data import Dataset, DataLoader

from prodigyopt import Prodigy

import trident

device = torch.device('cuda:0')

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class LinearModel(nn.Module):
    def __init__(self, input_shape):
        super(LinearModel, self).__init__()
        self.fc1 = trident.Linear(input_shape, 64)
        self.fc2 = trident.Linear(64, 64)
        self.fc3 = trident.Linear(64, 32)
        self.fc4 = trident.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

## main start
num_epoch = 500
batch_size = 64

data = load_breast_cancer()
x = data['data']
y = data['target']
#print(x.shape, y.shape)
#print(x[0], y[1])

train_dataset = dataset(x, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


model = LinearModel(input_shape=x.shape[1]).to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = Prodigy(model.parameters(), lr=1.0, use_bias_correction=True, weight_decay=0.1)
loss_func = nn.BCELoss()

start_time = time.time()
for i in range(num_epoch):
    avg_acc = 0
    avg_loss = 0
    for x_train, y_train in train_dataloader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        pred = model(x_train)
        loss = loss_func(pred, y_train.reshape(-1, 1))
        avg_acc += (pred.data.reshape(-1).to('cpu').numpy().round() == y_train.reshape(-1).to('cpu').numpy()).mean()
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 100 == 0:
        print(f'Epoch {i + 1} loss: {avg_loss:.2f}, acc: {avg_acc/len(train_dataloader):.2f}')
end_time = time.time()

print(f'Elapsed Time: {end_time - start_time:.2f} sec')
