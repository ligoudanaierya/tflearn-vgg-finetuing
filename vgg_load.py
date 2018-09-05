from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader
from tflearn.optimizers import Momentum
import h5py
import numpy as np 
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
import csv

tflearn.config.init_graph (num_cores=4, gpu_memory_fraction=0.5)

inp = input_data(shape=[None, 224, 224, 3])

conv1_1 = conv_2d(inp, 64, 3, activation='relu', name="conv1_1")
conv1_2 = conv_2d(conv1_1, 64, 3, activation='relu', name="conv1_2")
pool1 = max_pool_2d(conv1_2, 2, strides=2)

conv2_1 = conv_2d(pool1, 128, 3, activation='relu', name="conv2_1")
conv2_2 = conv_2d(conv2_1, 128, 3, activation='relu', name= "conv2_2")
pool2 = max_pool_2d(conv2_2, 2, strides=2)

conv3_1 = conv_2d(pool2, 256, 3, activation='relu', name="conv3_1")
conv3_2 = conv_2d(conv3_1, 256, 3, activation='relu', name="conv3_2")
conv3_3 = conv_2d(conv3_2, 256, 3, activation='relu', name="conv3_3")
pool3 = max_pool_2d(conv3_3, 2, strides=2)

conv4_1 = conv_2d(pool3, 512, 3, activation='relu', name="conv4_1")
conv4_2 = conv_2d(conv4_1, 512, 3, activation='relu', name="conv4_2")
conv4_3 = conv_2d(conv4_2, 512, 3, activation='relu', name="conv4_3")
pool4 = max_pool_2d(conv4_3, 2, strides=2)

conv5_1 = conv_2d(pool4, 512, 3, activation='relu', name="conv5_1")
conv5_2 = conv_2d(conv5_1, 512, 3, activation='relu', name="conv5_2")
conv5_3 = conv_2d(conv5_2, 512, 3, activation='relu', name="conv5_3")
pool5 = max_pool_2d(conv5_3, 2, strides=2)

fc6 = fully_connected(pool5, 4096, activation='relu', name="fc6")
fc6_dropout = dropout(fc6, 0.5)

fc7 = fully_connected(fc6_dropout, 4096, activation='relu', name="fc7")
fc7_droptout = dropout(fc7, 0.5)

fc8 = fully_connected(fc7_droptout, 1000, activation='softmax', name="fc8")

mm = Momentum(learning_rate=0.01, momentum=0.9, lr_decay=0.1, decay_step=1000)

network = regression(fc8, optimizer=mm, loss='categorical_crossentropy', restore=False)

model = tflearn.DNN(network)
model.load("/disk1/featureGroup/lsy/tflearn/vgg16_weights.tfl")
