# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10000)
y = 1 / (1 + np.exp(-x))
y1 = (1 - np.exp(-x)) / (1 + np.exp(-x))
# y1 = -np.log(1 / x - 1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(x, y, color='m', linestyle='-', marker='', label='f1')
plt.plot(x, y1, color='g', linestyle='-', marker='', label='f2')
# plt.plot(x, y1, color='b', linestyle='-', marker='', label='a = {}'.format(a))
plt.legend(loc='best')
plt.show()
