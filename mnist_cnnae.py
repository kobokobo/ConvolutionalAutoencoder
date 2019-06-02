#
#   mnist_cnnae.py   data 2019.May.30
#   
#   Autoencoder tutorial code - trial of convolutional AE
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer

# Up-sampling 2-D Layer (deconvolutoinal Layer)
class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''
    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, flg_batch_norm, activation='relu'):
        self.input = input      
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.out_ch = out_ch
        self.activation = activation
        self.flg_batch_norm = flg_batch_norm

        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]    # note the arguments order
        
        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
                            trainable=True)
        b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
                            trainable=True)
        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]      
        linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                            strides=[1, 2, 2, 1], padding='SAME') + self.b

        batch_norm = tf.layers.batch_normalization(linout, training=self.flg_batch_norm)
        if self.activation == 'relu':
            self.output = tf.nn.relu(batch_norm)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(batch_norm)
        else:
            self.output = batch_norm
        
        return self.output

# Create the model
def model(X, w_e, b_e, w_d, b_d):
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)
    
    return encoded, decoded

def mk_nn_model(x, y_,is_training):
    # Encoding phase
    x_image = tf.reshape(x, [-1, 28, 28, 1])    
    conv1 = Convolution2D(x_image, (28, 28), 1, 16, (3, 3), is_training, activation='relu')
    conv1_out = conv1.output()
    
    pool1 = MaxPooling2D(conv1_out)
    pool1_out = pool1.output()
    
    conv2 = Convolution2D(pool1_out, (14, 14), 16, 8, (3, 3), is_training, activation='relu')
    conv2_out = conv2.output()
    
    pool2 = MaxPooling2D(conv2_out)
    pool2_out = pool2.output()

    conv3 = Convolution2D(pool2_out, (7, 7), 8, 8, (3, 3), is_training, activation='relu')
    conv3_out = conv3.output()

    pool3 = MaxPooling2D(conv3_out)
    pool3_out = pool3.output()
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    # Decoding phase
    conv_t1 = Conv2Dtranspose(pool3_out, (7, 7), 8, 8, (3, 3), is_training, activation='relu')
    conv_t1_out = conv_t1.output()

    conv_t2 = Conv2Dtranspose(conv_t1_out, (14, 14), 8, 8,(3, 3), is_training, activation='relu')
    conv_t2_out = conv_t2.output()

    conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 8, 16, (3, 3), is_training, activation='relu')
    conv_t3_out = conv_t3.output()

    conv_last = Convolution2D(conv_t3_out, (28, 28), 16, 1, (3, 3), is_training, activation='sigmoid')
    decoded = conv_last.output()

    decoded = tf.reshape(decoded, [-1, 784])
    cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)

    return loss, decoded


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(dtype=tf.bool)

    loss, decoded = mk_nn_model(x, y_, is_training)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    init = tf.initialize_all_variables()
    # Train
    with tf.Session() as sess:
        sess.run(init)
        print('Training...')
        for i in range(10001):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            train_step.run({x: batch_xs, y_: batch_ys, is_training: True})
            if i % 1000 == 0:
                train_loss= loss.eval({x: batch_xs, y_: batch_ys, is_training: True})
                print('  step, loss = %6d: %6.3f' % (i, train_loss))

        # generate decoded image with test data
        test_fd = {x: mnist.test.images, y_: mnist.test.labels, is_training: True}
        decoded_imgs = decoded.eval(test_fd)
        print('loss (test) = ', loss.eval(test_fd))
     
    x_test = mnist.test.images
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    #plt.show()
    plt.savefig('mnist_cae.png')

