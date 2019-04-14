# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

# 定义三维数据
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)

X, Y = np.meshgrid(x, y)

pair = np.empty(shape=[0, 2], dtype=float)

# for i in range(len(x)):
#     for j in range(len(y)):
#         pair = np.append(pair, [[x[i], y[j]]], axis=0)
# print(pair)

Z = 10 / ((4 * np.square(X) - 2.1 * np.power(X, 4) + (1 / 3) * np.power(X, 6) +
           X * Y - 4 * np.square(Y) + 4 * np.power(Y, 4) + 2) * 11)
# print(Z)

# label = np.empty(shape=[0, 1], dtype=float)
# for i in range(len(x)):
#     for j in range(len(y)):
#         label = np.append(label, [[Z[j, i]]], axis=0)

# print(label)

# 定义坐标轴
fig = plt.figure()
ax3 = plt.axes(projection='3d')

# 作图
surf = ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet')
# ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 1), nn.Sigmoid())

        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.layer(x)
        return x


pair_t = torch.from_numpy(pair).float()
label_t = torch.from_numpy(label).float()
net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True)

# in your training loop:
# for i in range(6):
#     for j in range(10000):
#         optimizer.zero_grad()
#         out = net(pair_t[j])
#         loss = criterion(out, label_t[j])
#         loss.backward()
#         optimizer.step()
