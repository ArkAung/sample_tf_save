import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])  # Reshape input image to a 4-D Tensor

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name="conv-1") + b_conv1)  # Convolve and then ReLU
h_pool1 = max_pool_2x2(h_conv1, name="pool-1")  # Down sampling from 28x28 to 14x14

W_conv2 = weight_variable([5, 5, 32, 64])  #
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name="conv-2") + b_conv2)  # Convolve and then ReLU
h_pool2 = max_pool_2x2(h_conv2, name="pool-2")  # Down sampling from 14x14 to 7x7

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

'''
Accuracy
'''
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables
saver = tf.train.Saver()

costs_train = []
step = 200

for i in range(step):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                  y_: batch[1],
                                                  keep_prob: 1.0})
        print("Step {}, training accuracy {}".format(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0],
                              y_: batch[1],
                              keep_prob: 0.5})
    train_loss = cross_entropy.eval(feed_dict={x: batch[0],
                                               y_: batch[1],
                                               keep_prob: 1.0})
    costs_train.append(train_loss)

'''
Saving weights as sample for conv1 weights and biases
'''
# Evaluate weights and biases to be saved
weights_conv1 = W_conv1.eval()
biases_conv1 = b_conv1.eval()

# Save as npy arrays
np.save('weights-conv1.npy', weights_conv1)
np.save('biases-conv1.npy', biases_conv1)

# Save the graph
saver.save(sess, 'conv_MNIST_model', global_step=step)

print("Test Accuracy: {}".format(accuracy.eval(feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,
                                                         keep_prob: 1.0})))
print("Test Cost: {}".format(cross_entropy.eval(feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,
                                                         keep_prob: 1.0})))