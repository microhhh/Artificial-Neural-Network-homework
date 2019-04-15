# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

# %matplotlib inline

n = 1000
# 定义三维数据
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)

X, Y = np.meshgrid(x, y)

pair = np.zeros(shape=[n * n, 2], dtype=float)

k = 0
for i in range(len(x)):
    for j in range(len(y)):
        pair[k, 0] = x[i]
        pair[k, 1] = y[j]
        k += 1

# print(pair)
print(len(pair))

Z = 10 / ((4 * np.square(X) - 2.1 * np.power(X, 4) + (1 / 3) * np.power(X, 6) +
           X * Y - 4 * np.square(Y) + 4 * np.power(Y, 4) + 2) * 11)
# print(X)
# print(Y)
# print(Z)

label = np.zeros(shape=[n * n, 1], dtype=float)
k = 0
for i in range(len(x)):
    for j in range(len(y)):
        label[k, 0] = Z[j, i]
        k += 1

# print(label)
print(len(label))


# 绘图
# fig = plt.figure()
# ax3 = plt.axes(projection='3d')

# surf = ax3.plot_surface(X, Y, Z, cstride=1, cmap='jet')
# plt.contour(X, Y, Z, 90, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
# fig.colorbar(surf)
# plt.show()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        # self.layer = nn.Sequential(nn.Linear(2, 7), nn.Sigmoid(), nn.Linear(7, 1))

        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.layer(x)
        return x


pair_t = torch.from_numpy(pair).float().cuda()
label_t = torch.from_numpy(label).float().cuda()
net = Net().cuda()

# print(pair_t)
# print(label_t)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(net.parameters(), lr=0.03)

# batch_size = 1000
# for i in range(100):
#     for j in range(0, n * n, batch_size):
#         optimizer.zero_grad()
#         out = net(pair_t[j:j + batch_size])
#         loss = criterion(out, label_t[j:j + batch_size])
#         loss.backward()
#         optimizer.step()
#     print(loss.item())

for i in range(5000):
    optimizer.zero_grad()
    out = net(pair_t)
    # print(out)
    loss = criterion(out, label_t)
    loss.backward()
    optimizer.step()
    print(loss.item())

net.eval()
out = net(pair_t)

z_out = out.cpu().detach().numpy()
print(z_out.shape)

Z_out = np.zeros(shape=[n, n], dtype=float)
k = 0
for i in range(len(x)):
    for j in range(len(y)):
        Z_out[j, i] = z_out[k, 0]
        k += 1

print(list(net.named_parameters()))

# 绘图
fig = plt.figure()
ax3 = plt.axes(projection='3d')

surf = ax3.plot_surface(X, Y, Z_out, cstride=1, cmap='jet')
# plt.contour(X, Y, Z, 90, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
fig.colorbar(surf)
plt.show()
