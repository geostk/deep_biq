from sklearn import datasets

import tensorflow as tf
import numpy as np

(x_vals, y_vals) = datasets.make_regression(n_samples=1000, n_features=1, noise=10)

train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]


def score(X):
    l2 = tf.contrib.layers.l2_regularizer(0.0005)
    W = tf.get_variable(name='W', initializer=tf.truncated_normal_initializer(stddev=1.0, mean=0, dtype=tf.float32),
                        shape=[1])
    bias = tf.get_variable(name='b', shape=[1], initializer=tf.zeros_initializer(dtype=tf.float32))
    score = tf.matmul(W, X) + bias
    return score


def loss(y, y1):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(y, y1)), name='loss')
    return loss


X = tf.placeholder(dtype=tf.float32, shape=[None, 1])

y = tf.placeholder(dtype=tf.float32, shape=[None])
global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)
score = score(X)
loss = loss(score, y)
lr = tf.train.exponential_decay(0.0001,
                                global_step,
                                10000,
                                0.16,
                                staircase=True)
opt = tf.train.GradientDescentOptimizer(lr)
train_op = opt.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
train_loss = []
for i in range(5000):
    rand_index = np.random.choice(len(x_vals_train), size=32)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    learning_rate, _, temp_loss = sess.run([lr, train_op, loss], feed_dict={X: rand_x, y: rand_y})
    # temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    train_loss.append(temp_loss)
    if i % 100 == 0:
        print('step #' + str(i))
        print('loss #' + str(temp_loss))
        print('lr #' + str(learning_rate))
