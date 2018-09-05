import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import tensorflow as tf
import numpy as np

datadict = np.load('vgg16_weights.npz')
print(datadict['convl_1'][0].shape)

def vgg16(input, num_class):
    
    x = tflearn.conv_2d(input, 64, 3, activation='relu',scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, scope='maxpool1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, scope='maxpool2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, scope='maxpool3')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, scope='maxpool4')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, scope='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')
    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')
    x = tflearn.fully_connected(x, num_class, activation='softmax',scope='fc8', restore=False)
    return x

data_dir = '/path/to/images'
model_path = '/path/to/vggmodel'
data_list = 'path/to/list'

from tflearn.data_utils import image_preloader
X, Y = image_preloader(data_list, image_shape=(224,224), mode='file',
                        categorical_labels=True, normalize=False, 
                        files_extension=['.jpg','.png'],filter_channel=True)
'''
X, Y = image_preloader(data_dir,image_shape=(224,224), mode='folder',
                        categorical_labels=True, normalize=False,
                        files_extension=['.jpg','.png'],filter_channel=True)
'''
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68,116.779,103.939],per_channel=True)

x = tflearn.input_data(shape=[None,224,224,3], name='input', data_preprocessing=img_prep)

softmax = vgg16(x, num_class)
regression = tflearn.regression(softmax, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001, restore=False)
model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning', max_checkpoints=3, tensorboard_verbose=2, tensorboard_dir='./logs')
model_file = os.path.join(model_payh, "vgg16.tflearn")
model.load(model_file, weights_only=True)
model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=Ture, show_metric=True, batch_size=64, snapshot_epoch=False, snapshot_step=200,
            run_id='vgg-finetuing')
model.save('my_vgg_model')

