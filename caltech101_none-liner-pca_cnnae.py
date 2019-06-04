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

from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer
from my_nn_lib import FullConnected_sigmoid_upscale



bit_length = 15
pncode0 = np.array([-1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1], dtype = 'float')
pncode1 = np.array([-1,  -1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1], dtype = 'float')
pncode2 = np.array([-1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1], dtype = 'float')
pncode3 = np.array([ 1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1], dtype = 'float')
pncode4 = np.array([-1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1], dtype = 'float')
pncode5 = np.array([-1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1], dtype = 'float')
pncode6 = np.array([ 1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1], dtype = 'float')
pncode7 = np.array([ 1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1], dtype = 'float')
pncode8 = np.array([-1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1,   1], dtype = 'float')
pncode9 = np.array([ 1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1], dtype = 'float')



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
    with tf.variable_scope('cae'):
        # Encoding phase
        x_image = tf.reshape(x, [-1, 32*32*3])
        conv1 = Convolution2D(x, (32, 32), 3, 64, (3, 3), is_training, activation='relu')
        conv1_out = conv1.output()

        pool1 = MaxPooling2D(conv1_out)
        pool1_out = pool1.output()
        pool1_out_flat = tf.reshape(pool1_out, [-1, 16*16*64])
        # representation is (16, 16, 64) i.e. 16384-dimensional

        conv2 = Convolution2D(pool1_out, (16, 16), 64, 32, (3, 3), is_training, activation='relu')
        conv2_out = conv2.output()

        pool2 = MaxPooling2D(conv2_out)
        pool2_out = pool2.output()
        pool2_out_flat = tf.reshape(pool2_out, [-1, 8*8*32])
        # representation is (8, 8, 32) i.e. 2048-dimensional

        conv3 = Convolution2D(pool2_out, (8, 8), 32, 16, (3, 3), is_training, activation='relu')
        conv3_out = conv3.output()

        pool3 = MaxPooling2D(conv3_out)
        pool3_out = pool3.output()
        pool3_out_flat = tf.reshape(pool3_out, [-1, 4*4*16])
        # representation is (4, 4, 16) i.e. 256-dimensional

        conv_t1 = Conv2Dtranspose(pool3_out, (8, 8), 16, 16, (3, 3), is_training, activation='relu')
        conv_t1_out = conv_t1.output()

        conv_t2 = Conv2Dtranspose(conv_t1_out, (16, 16), 16, 32, (3, 3), is_training, activation='relu')
        conv_t2_out = conv_t2.output()

        conv_t3 = Conv2Dtranspose(conv_t2_out, (32, 32), 32, 64, (3, 3), is_training, activation='relu')
        conv_t3_out = conv_t3.output()

        conv_last = Convolution2D(conv_t3_out, (32, 32), 64, 3, (3, 3), is_training, activation='sigmoid')
        decoded = conv_last.output()

        decoded = tf.reshape(decoded, [-1, 32*32*3])
        p = tf.log(tf.clip_by_value(decoded, 1e-10, 1.0))
        q = tf.log(tf.clip_by_value((1. - decoded), 1e-10, 1.0))
        cross_entropy = -1. * x_image * p - (1. - x_image) * q
        loss = tf.reduce_mean(cross_entropy)

        feature_descriptor = tf.concat([pool1_out_flat, pool2_out_flat, pool3_out_flat], 1)

        return loss, decoded, feature_descriptor


def mk_nn_model2(rep, y_, is_training):
    with tf.variable_scope('fc-pn'):


        fc1 = FullConnected_sigmoid_upscale(rep, (1, 18688), 1, 15, (3, 3), is_training, activation='relu')

        '''
        # Encoding phase
        x_image = tf.reshape(x, [-1, 32 * 32 * 3])
        conv1 = Convolution2D(x, (32, 32), 3, 64, (3, 3), is_training, activation='relu')
        conv1_out = conv1.output()

        pool1 = MaxPooling2D(conv1_out)
        pool1_out = pool1.output()
        # representation is (16, 16, 64) i.e. 16384-dimensional

        conv2 = Convolution2D(pool1_out, (16, 16), 64, 32, (3, 3), is_training, activation='relu')
        conv2_out = conv2.output()

        pool2 = MaxPooling2D(conv2_out)
        pool2_out = pool2.output()
        # representation is (8, 8, 32) i.e. 2048-dimensional

        conv3 = Convolution2D(pool2_out, (8, 8), 32, 16, (3, 3), is_training, activation='relu')
        conv3_out = conv3.output()

        pool3 = MaxPooling2D(conv3_out)
        pool3_out = pool3.output()
        # representation is (4, 4, 16) i.e. 256-dimensional

        pool3_out_update = tf.concat([pool3_out, rep], 3)
        pool3_out_update = tf.reshape(pool3_out_update, [-1, 4, 4, 32])

        conv_t1 = Conv2Dtranspose(pool3_out_update, (8, 8), 32, 16, (3, 3), is_training, activation='relu')
        conv_t1_out = conv_t1.output()

        conv_t2 = Conv2Dtranspose(conv_t1_out, (16, 16), 16, 32, (3, 3), is_training, activation='relu')
        conv_t2_out = conv_t2.output()

        conv_t3 = Conv2Dtranspose(conv_t2_out, (32, 32), 32, 64, (3, 3), is_training, activation='relu')
        conv_t3_out = conv_t3.output()

        conv_last = Convolution2D(conv_t3_out, (32, 32), 64, 3, (3, 3), is_training, activation='sigmoid')
        decoded = conv_last.output()

        decoded = tf.reshape(decoded, [-1, 32 * 32 * 3])
        p = tf.log(tf.clip_by_value(decoded, 1e-10, 1.0))
        q = tf.log(tf.clip_by_value((1. - decoded), 1e-10, 1.0))
        cross_entropy = -1. * x_image * p - (1. - x_image) * q
        loss = tf.reduce_mean(cross_entropy)
        '''
        return loss, decoded


def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



# setting of feed_dict (parameter)
def set_feed(images, labels, prob, pn):

    pncode = np.zeros((0, bit_length))
    for i in range(labels.shape[0]):

        if (labels[i,:] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode0.reshape(1,-1)]
        elif (labels[i,:] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode1.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode2.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode3.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode4.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode5.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 1 , 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode6.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
            pncode = np.r_[pncode, pncode7.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all():
            pncode = np.r_[pncode, pncode8.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
            pncode = np.r_[pncode, pncode9.reshape(1,-1)]

    # print(pncode.shape)
    return {x: images, y_: pncode, y_org : labels, keep_prob: prob, pn_: pn}


if __name__ == '__main__':

    nums = 10
    X_train, X_test, Y_train, Y_test = np.load("/data/101_Caltech_npy/32-32img-10obj.npy")
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256

    # Variables
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="x")  # image data
    y_ = tf.placeholder(tf.float32, shape=(None, nums), name="y_")  # label (teaching data)
    is_training = tf.placeholder(dtype=tf.bool)
    rep_ = tf.placeholder(tf.float32, shape=(None, 18688), name="rep_")  # image data


    # Optimise
    loss, decoded, feature_rep = mk_nn_model(x, y_, is_training)
    optim_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cae")
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=optim_vars1)



    '''
    loss2, decoded2 = mk_nn_model2(x, y_, rep_, is_training)
    optim_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pca2")
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(loss2, var_list=optim_vars2)
    #train_step2 = tf.train.AdamOptimizer(1e-3).minimize(loss2)
    '''

    # init
    init = tf.initialize_all_variables()

    # Train1
    epochs =100
    with tf.Session() as sess:
        sess.run(init)
        print('Training...')
        for j in range(epochs):
            for i in range(int(len(X_train)/32)):
                batch_xs, batch_ys = next_batch(32, X_train, Y_train)
                train_step.run({x: batch_xs, y_: batch_ys, is_training: True})
            if j % 1 == 0:
                train_loss = loss.eval({x: batch_xs, y_: batch_ys, is_training: True})
                print('  epochs, loss = %6d: %6.3f' % (j, train_loss))


        # generate decoded image with test data
        fd = {x: batch_xs, y_: batch_ys, is_training: True}
        decoded_imgs = decoded.eval(fd)
        print('loss (test) = ', loss.eval(fd))


        x_test = batch_xs
        y_test = batch_ys
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(32, 32, 3))

            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        #plt.show()
        plt.savefig('caltech101_cae_first.png')


        # start training
        for step in range(10000):

            batch_xs, batch_ys = next_batch(32, X_train, Y_train)
            rep_temp = train_step.eval({x: batch_xs, y_: batch_ys, is_training: True})

            batch_xs, batch_ys = next_batch(32, X_train, Y_train)
            fd = set_feed(batch_xs[0], batch_ys[1], 0.5, pn_all)

            _, loss = sess.run([train_step, cross_entoropy], feed_dict=fd)
            # _, loss, temp = sess.run([train_step, cross_entoropy, W_fc3], feed_dict=fd)
            # print("xxxxx debug xxxxx", temp)

            if step % 100 == 0:
                acc, w_summary = sess.run([accuracy_step, summary], feed_dict=test_fd)
                print("step=", step, "loss=", loss, "acc=", acc)
                tw.add_summary(w_summary, step)


        pn_all = np.c_[pncode0, pncode1, pncode2, pncode3, pncode4, pncode5, pncode6, pncode7, pncode8, pncode9]
        test_fd = set_feed(mnist.test.images, mnist.test.labels, 1, pn_all)


        '''
        # Train2
        for j in range(epochs):
            for i in range(int(len(X_train) / 32)):
                batch_xs, batch_ys = next_batch(32, X_train, Y_train)
                rep_temp = first_rep.eval({x: batch_xs, y_: batch_ys, is_training: True})
                sess.run(train_step2, feed_dict={x: batch_xs, y_: batch_ys, rep_:rep_temp, is_training: True})
            if j % 1 == 0:
                train_loss2 = loss2.eval({x: batch_xs, y_: batch_ys, rep_:rep_temp, is_training: True})
                print('  epochs, loss = %6d: %6.3f' % (j, train_loss2))


        # generate decoded image with test data
        rep_temp = first_rep.eval(fd)
        decoded_imgs2 = decoded2.eval({x: x_test, y_: y_test, rep_:rep_temp, is_training: True})
        print('loss (test) = ', loss2.eval({x: x_test, y_: y_test, rep_:rep_temp, is_training: True}))

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(32, 32, 3))

            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs2[i].reshape(32, 32, 3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig('caltech101_cae_second.png')
        '''