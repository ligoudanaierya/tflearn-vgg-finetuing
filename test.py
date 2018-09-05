import tflearn
import matplotlib.pyplot as plt
import h5py

h5f = h5py.File('dataset.h5','r')
X = h5f['X']
Y = h5f['Y']

plt.imshow(X[0])
plt.show
