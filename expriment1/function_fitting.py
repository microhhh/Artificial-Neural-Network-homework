# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

n = 1000
# 定义三维数据
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)

X, Y = np.meshgrid(x, y)

test_n = 150
x_test = np.linspace(-3, 3, test_n)
y_test = np.linspace(-3, 3, test_n)

X_test, Y_test = np.meshgrid(x_test, y_test)

pair = np.zeros(shape=[n * n, 2], dtype=float)
pair_t = np.zeros(shape=[test_n * test_n, 2], dtype=float)

k = 0
for i in range(len(x)):
    for j in range(len(y)):
        pair[k, 0] = x[i]
        pair[k, 1] = y[j]
        k += 1

print(len(pair))

k = 0
for i in range(len(x_test)):
    for j in range(len(y_test)):
        pair_t[k, 0] = x_test[i]
        pair_t[k, 1] = y_test[j]
        k += 1

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

print(len(label))


# 随机化数据
# def shuffle():
#     for i in range(100000):
#         rand_1 = np.random.randint(low=0, high=n * n)
#         rand_2 = np.random.randint(low=0, high=n * n)
#         swap_pair = pair[rand_1]
#         swap_label = label[rand_1]
#         pair[rand_1] = pair[rand_2]
#         label[rand_1] = label[rand_2]
#         pair[rand_2] = swap_pair
#         label[rand_2] = swap_label


# 绘图
# fig = plt.figure()
# # ax3 = plt.axes(projection='3d')
#
# # surf = ax3.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap='jet')
# plt.contour(X, Y, Z, 90, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
# # fig.colorbar(surf)
# plt.show()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        # self.layer = nn.Sequential(nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 32),
        #                            nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 1))
        # self.layer = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        # self.layer = nn.Sequential(nn.Linear(2, 16), nn.Sigmoid(), nn.Linear(16, 32), nn.Sigmoid(), nn.Linear(32, 1))
        # self.layer = nn.Sequential(nn.Linear(2, 7), nn.ReLU(), nn.Linear(7, 1))

        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight)

    def forward(self, x):
        x = self.layer(x)
        return x


pair_train = torch.from_numpy(pair).float().to(device)
label_train = torch.from_numpy(label).float().to(device)

pair_test = torch.from_numpy(pair_t).float().to(device)

net = Net().to(device)

# print(pair_t)
# print(label_t)

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.4, nesterov=True)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
optimizer = optim.Adam(net.parameters(), lr=0.03)

batch_size = 10000
loss = 0

# for i in range(50):
#     # shuffle()
#     j = 0
#     while j < n * n:
#         optimizer.zero_grad()
#         out = net(pair_train[j:j + batch_size])
#         # print(out)
#         loss = criterion(out, label_train[j:j + batch_size])
#         print(loss.item())
#         loss.backward()
#         optimizer.step()
#         j += batch_size
#     print(loss.item())
#     scheduler.step()

for i in range(5000):
    optimizer.zero_grad()
    out = net(pair_train)
    # print(out)
    loss = criterion(out, label_train)
    loss.backward()
    optimizer.step()
    # print(loss.item())

print(loss.item())
net.eval()
out = net(pair_test)

z_out = out.cpu().detach().numpy()
print(z_out.shape)

Z_out = np.zeros(shape=[test_n, test_n], dtype=float)
k = 0
for i in range(len(x_test)):
    for j in range(len(y_test)):
        Z_out[j, i] = z_out[k, 0]
        k += 1

print(net)
print(list(net.named_parameters()))

# 绘图
fig = plt.figure()
ax3 = plt.axes(projection='3d')

surf = ax3.plot_surface(X_test, Y_test, Z_out, rstride=1, cstride=1, cmap='jet')
# plt.contour(X_test, Y_test, Z_out, 90, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
fig.colorbar(surf)

fig1 = plt.figure('fig1')
ax2 = plt.axes(projection=None)
plt.contour(X_test, Y_test, Z_out, 90, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
plt.show()
