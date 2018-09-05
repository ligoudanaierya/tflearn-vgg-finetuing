from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.data_utils import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

from tflearn.datasets import cifar10
(X, Y),(X_test, Y_test) = cifar10.load_data()
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

import  h5py

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('cifar10_X', data=X)
h5f.create_dataset('cifar10_Y', data=Y)
h5f.create_dataset('cifar10_X_test', data=X_test)
h5f.create_dataset('cifar10_Y_test', data=Y_test)
h5f.close()

h5f = h5py.File('data.h5','r')
X = h5f['cifar10_X']
Y = h5f['cifar10_Y']
X_test = h5f['cifar10_X_test']
Y_test = h5f['cifar10_Y_test']

network =  input_data(shape=[None,32,32,3], dtype=tf.float32)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network,0.5)

softmax = fully_connected(network,10, activation='softmax')
regression = regression(softmax, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.001)

model = tflearn.DNN(regression,tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test,Y_test), show_metric=True, batch_size=96, run_id='cifar10_cnn')
 
