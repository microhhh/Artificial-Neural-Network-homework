# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1, 1, 1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 0, 0, 1],
                  [1, 0, 0, 1, 0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0, 1, 1, 1, 1],
                  [1, 1, 1, 0, 0, 1, 0, 0, 1],
                  [1, 1, 1, 1, 0, 0, 1, 0, 0]], dtype=torch.float)
y = torch.tensor([[1],
                  [1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0],
                  [0]], dtype=torch.float)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer = nn.Sequential(nn.Linear(9, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())

        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.layer(x)
        return x


net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True)

# in your training loop:
for i in range(6):
    for j in range(8):
        optimizer.zero_grad()
        out = net(x[j])
        loss = criterion(out, y[j])
        loss.backward()
        optimizer.step()



print(list(net.named_parameters()))
out = net(x)
print(out)
