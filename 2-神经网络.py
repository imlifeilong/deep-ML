import numpy as np
import tensorflow as tf


def sigmoid(x):
    # 激活函数， 值域在[0, 1] x越大值接近1，x越小值接近0
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def _sigmoid(x):
    '''sigmoid导函数，可以用自身表示'''
    # f'(x) = f(x) * (1 - f(x))
    sx = sigmoid(x) * (1 - sigmoid(x))
    return sx


def loss(y_t, y_p):
    '''

    :param y_t: 真实值
    :param y_p: 预测值
    :return:
    '''
    # 方差的平均值
    return ((y_t - y_p) ** 2).mean()


class Neuron(object):
    '''神经元'''

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feed(self, input):
        # 矩阵乘法
        total = np.dot(self.w, input) + self.b

        return sigmoid(total)


class NeuronNet(object):
    '''神经网络'''

    def __init__(self):
        w = np.array([0, 1])
        b = 0
        # 隐藏层
        self.h1 = Neuron(w, b)
        self.h2 = Neuron(w, b)
        # 输出层
        self.o1 = Neuron(w, b)

    def feed(self, x):
        oh1 = self.h1.feed(x)
        oh2 = self.h2.feed(x)
        oo1 = self.o1.feed(np.array([oh1, oh2]))

        return oo1


class SelfNeuronNet(object):
    def __init__(self):
        # 服从正太分布的随机数
        # np.random.normal(loc=0.0, scale=1e-2)
        # loc 表示分布的中心，0表示以y轴为中心
        # scale 表示分布的宽度，越大表示矮胖，越小表示高廋
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feed(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1

    def train(self, data, y_trues):
        learn_rate = 0.1

        for i in range(1000):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * _sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * _sigmoid(sum_o1)
                d_ypred_d_b3 = _sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * _sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * _sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * _sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * _sigmoid(sum_h1)
                d_h1_d_b1 = _sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * _sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * _sigmoid(sum_h2)
                d_h2_d_b2 = _sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch
            if i % 10 == 0:
                y_preds = np.apply_along_axis(self.feed, 1, data)
                los = loss(y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (i, los))


if __name__ == '__main__':
    # print('#########神经元#########')
    # w = np.array([0, 1])
    # b = 4
    # n = Neuron(w, b)
    # x = np.array([2, 3])
    # print(n.feed(x))
    #
    # #########
    # print('#######神经网络##########')
    # nn = NeuronNet()
    # print(nn.feed(x))
    # ###########
    #
    # print('########损失函数#########')
    # y_t = np.array([1, 0, 0, 1])
    # y_p = np.array([0, 0, 0, 0])
    #
    # print(loss(y_t, y_p))

    s_data = np.array([
        [133, 65],
        [160, 72],
        [152, 70],
        [120, 60],
    ])


    gap = np.array([135, 66])

    # 对列求期望
    # avg = s_data.mean(axis=0)

    # data = s_data - gap

    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])

    y_trues = np.array([
        1, 0, 0, 1
    ])

    nt = SelfNeuronNet()
    nt.train(data, y_trues)

    e = np.array([-7, -3])
    f = np.array([20, 2])
    print(nt.feed(e), nt.feed(f))
