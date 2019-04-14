# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt

plt.figure()
plt.plot([0, 0], [-3, 3], label="x=0", linestyle='--', color="c")
plt.plot([-3, 3], [-1, 2], label="y=0.5x+0.5", linestyle='--', color="b")

# 定义新的坐标标注点
plt.xlim((-3, 3))
plt.ylim((-3, 3))
ax = plt.gca()  # get current axis
# 将右边的坐标线颜色取消
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 设置坐标系的中心， data对应的是y轴数据和x轴数据
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 标注点 x, y
x1 = [1, 1]
y1 = [1, 2]
x2 = [2, 2]
y2 = [-1, 0]
x3 = [-1, -2]
y3 = [2, 1]
x4 = [-1, -2]
y4 = [-1, -2]

plt.scatter(x1, y1, c='green')
plt.scatter(x2, y2, c='yellow')
plt.scatter(x3, y3, c='red')
plt.scatter(x4, y4, c='magenta')

offset = 2
plt.annotate(s="x1", xy=(x1[0], y1[0]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x2", xy=(x1[1], y1[1]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x3", xy=(x2[0], y2[0]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x4", xy=(x2[1], y2[1]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x5", xy=(x3[0], y3[0]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x6", xy=(x3[1], y3[1]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x7", xy=(x4[0], y4[0]), xytext=(offset, offset), textcoords='offset points')
plt.annotate(s="x8", xy=(x4[1], y4[1]), xytext=(offset, offset), textcoords='offset points')

plt.legend()
plt.show()
