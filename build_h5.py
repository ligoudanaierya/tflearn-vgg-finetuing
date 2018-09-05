import tensorflow as tf
import tflearn
import os 
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import build_hdf5_image_dataset
import h5py
data_dir = 'TU-Berlin'

build_hdf5_image_dataset(data_dir, image_shape=(224,224), mode='folder',output_path='dataset.h5', categorical_labels=True,normalize=True)

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

print(X.shape)
