# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-30, 30, 10000)
a = 1
y = 1 / (1 + np.exp(-0.1 * x))
y1 = 1 / (1 + np.exp(-1 * x))
y2 = 1 / (1 + np.exp(-10 * x))
# y1 = -np.log(1 / x - 1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(x, y, color='m', linestyle='-', marker='', label='a = {}'.format(0.1))
plt.plot(x, y1, color='g', linestyle='-', marker='', label='a = {}'.format(1))
plt.plot(x, y2, color='b', linestyle='-', marker='', label='a = {}'.format(10))
# plt.plot(x, y1, color='b', linestyle='-', marker='', label='a = {}'.format(a))
plt.legend(loc='best')
plt.show()
