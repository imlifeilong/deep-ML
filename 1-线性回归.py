'''
线性回归

1、随机误差项是一个期望值或平均值为0的随机变量；
2、对于解释变量的所有观测值，随机误差项有相同的方差；
3、随机误差项彼此不相关；
4、解释变量是确定性变量，不是随机变量，与随机误差项彼此之间相互独立；
5、解释变量之间不存在精确的（完全的）线性关系，即解释变量的样本观测值矩阵是满秩矩阵；
6、随机误差项服从正态分布。	

'''




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

v = []
for i in range(100):
    # 生成基于正太分布的随机数，服从均值为0，标准差为0.55
    x1 = np.random.normal(0.0, 0.55)
    # 生成在斜率为0.1, 偏置为0.3的线附近的点
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.05)
    v.append((x1, y1))

x_data = [j[0] for j in v]
y_data = [j[1] for j in v]

# w = tf.Variable(tf.random_uniform([1], -1, 1))
# 设置预测权重和参数
w = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
# # 定义模型 y 表示预测
y = tf.add(tf.multiply(w, x_data), b)
# 求损失函数
loss = tf.reduce_mean(tf.square(y-y_data))

# 梯度下降，学习率为0.1
# # 优化函数 梯度下降 0.0001表示学习率，越小收敛越慢
# # 梯度 函数某一点的导数，该点变化率
# # 梯度下降 沿梯度下降的方向求解极小值
# # 学习率 每次进行训练时在最陡的梯度方向上所采取的「步」长
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 优化函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(x_data, y_data, c='r')

with tf.Session() as sess:
    sess.run(init)
    print('w', sess.run(w), 'b', sess.run(b), 'loss', sess.run(loss))

    for i in range(200):
        sess.run(train)
        print('w', sess.run(w), 'b', sess.run(b), 'loss', sess.run(loss))
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        lines = ax.plot(x_data, sess.run(w)*x_data+sess.run(b))

        plt.pause(0.01)
    plt.show()
