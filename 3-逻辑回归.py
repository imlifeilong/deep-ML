
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


actv = tf.nn.softmax(tf.matmul(x, w)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))

rate = 0.01

opt = tf.train.GradientDescentOptimizer(rate).minimize(cost)

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(pred, "float"))


init = tf.global_variables_initializer()


fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(x_data, y_data, c='r')

with tf.InteractiveSession() as sess:
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