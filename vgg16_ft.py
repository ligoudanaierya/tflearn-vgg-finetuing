import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import h5py
import tensorflow as tf
import numpy as np
from  tflearn.datasets import oxflower17
#X, Y = oxflower17.load_data(one_hot=True)

datadict = np.load('vgg16_weights.npz')
#print(datadict.keys())
#print(datadict['conv1_1_W'].shape)
conv1_1 = tf.constant(datadict['conv1_1_W'])
bias1_1 = tf.constant(datadict['conv1_1_b'])
conv1_2 = tf.constant(datadict['conv1_2_W'])
bias1_2 = tf.constant(datadict['conv1_1_b'])
conv2_1 = tf.constant(datadict['conv2_1_W'])
bias2_1 = tf.constant(datadict['conv2_1_b'])
conv2_2 = tf.constant(datadict['conv2_2_W'])
bias2_2 = tf.constant(datadict['conv2_2_b'])
conv3_1 = tf.constant(datadict['conv3_1_W'])
bias3_1 = tf.constant(datadict['conv3_1_b'])
conv3_2 = tf.constant(datadict['conv3_2_W'])
bias3_2 = tf.constant(datadict['conv3_2_b'])
conv3_3 = tf.constant(datadict['conv3_3_W'])
bias3_3 = tf.constant(datadict['conv3_3_b'])
conv4_1 = tf.constant(datadict['conv4_1_W'])
bias4_1 = tf.constant(datadict['conv4_1_b'])
conv4_2 = tf.constant(datadict['conv4_2_W'])
bias4_2 = tf.constant(datadict['conv4_2_b'])
conv4_3 = tf.constant(datadict['conv4_3_W'])
bias4_3 = tf.constant(datadict['conv4_3_b'])
conv5_1 = tf.constant(datadict['conv5_1_W'])
bias5_1 = tf.constant(datadict['conv5_1_b'])
conv5_2 = tf.constant(datadict['conv5_2_W'])
bias5_2 = tf.constant(datadict['conv5_2_b'])
conv5_3 = tf.constant(datadict['conv5_3_W'])
bias5_3 = tf.constant(datadict['conv5_3_b'])
fc6_w = tf.constant(datadict['fc6_W'])
fc6_b = tf.constant(datadict['fc6_b'])
fc7_w = tf.constant(datadict['fc7_W'])
fc7_b = tf.constant(datadict['fc7_b'])

h5f = h5py.File('dataset.h5','r')
X = h5f['X']
Y = h5f['Y']
#print(Y[0])
X, Y = tflearn.data_utils.shuffle(X, Y)

def vgg16(input, num_class):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', weights_init=conv1_1,bias_init=bias1_1, name='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu',weights_init=conv1_2,bias_init=bias1_2, name='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2)
    x = tflearn.conv_2d(x, 128, 3, activation='relu',weights_init=conv2_1,bias_init=bias2_1, name='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu',weights_init=conv2_2,bias_init=bias2_2, name='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2)
    x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_1,bias_init=bias3_1, name='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_2,bias_init=bias3_2, name='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_3,bias_init=bias3_3, name='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2)
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_1,bias_init=bias4_1, name='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_2,bias_init=bias4_2, name='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_3,bias_init=bias4_3, name='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2)
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_1,bias_init=bias5_1, name='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_2,bias_init=bias5_2, name='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_3,bias_init=bias5_3, name='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2)
                                                                            
    x = tflearn.fully_connected(x, 4096, activation='relu',weights_init=fc6_w,bias_init=fc6_b, name='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')
    x = tflearn.fully_connected(x, 4096, activation='relu',weights_init=fc7_w,bias_init=fc7_b, name='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')
    x = tflearn.fully_connected(x, num_class, activation='softmax',name='fc8', restore=False)
    return x
#img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center(mean=[123.68,116.779,103.939],per_channel=True)
#x = tflearn.input_data(shape=[None,224,224,3], name='input',data_preprocessing=img_prep)
x = tflearn.input_data(shape=[None,224,224,3], name='input')

softmax = vgg16(x,250)
regression = tflearn.regression(softmax, optimizer='rmsprop', loss = 'categorical_crossentropy', learning_rate=0.0001)
model = tflearn.DNN(regression,checkpoint_path='vgg-finetuing/vgg-ft',max_checkpoints=3, tensorboard_verbose=2, tensorboard_dir='./logs')
vars = tflearn.variables.get_layer_variables_by_name('conv1_1')
print(type(model.get_weights(vars[0])))
vars = tflearn.variables.get_layer_variables_by_name('conv1_2')
print(type(model.get_weights(vars[0])))

#model.load("vgg16_weights.tflearn")
model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True, show_metric=True, batch_size=32, snapshot_epoch=False,
                snapshot_step=500, run_id='vgg-finetuing')
model.save('TU-vgg.tfl')
