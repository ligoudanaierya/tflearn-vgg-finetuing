from __future__ import absolute_import, division, print_function

import tflearn

X=[[0.],[40], [120.], [250.]]
Y = [[0.],[12.],[24.],[38.] ]

network = tflearn.input_data(shape=[None,1])
network = tflearn.fully_connected(network,32,activation='linear')
network = tflearn.fully_connected(network, 32, activation='linear')
network = tflearn.fully_connected(network, 1, activation='sigmoid')
network = tflearn.regression(network, optimizer='sgd',learning_rate=0.1, loss='mean_square')
m = tflearn.DNN(network)
m.fit(X, Y, n_epoch=1000, snapshot_epoch=False)
print(m.predict([[80.]]))
