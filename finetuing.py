from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

#X, Y = your_dataset()

num_classes = 10

network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3,activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network,0.75)
network = conv_2d(network, 64,3,activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network,512, activation='relu')
network = dropout(network, 0.5)
softmax = fully_connected(network, num_classes,activation='softmax',restore=False)
regression = regression(softmax,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
model = tflearn.DNN(regression, checkpoint_path='model_finetuning',max_checkpoints=3,tensorboard_verbose=0)

model.load('cifar10_cnn')
model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_step=200,
            snapshot_epoch=False, run_id='model_finetuning')

