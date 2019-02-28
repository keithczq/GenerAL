from __future__ import print_function

import os

import numpy as np

from utils import np_to_tfrecords

dir = os.path.expanduser('~/Desktop/critic_dataset/')

prefixs = []
prefix_to_x_filename = {}
prefix_to_y_filename = {}
for i in os.listdir(dir):
    if i.endswith('npy'):
        prefix = str(i).split('_')[0][:-2]
        prefixs.append(prefix)
        if '_x' in i:
            prefix_to_x_filename[prefix] = i
        elif '_y' in i:
            prefix_to_y_filename[prefix] = i

prefixs = set(prefixs)

X, Y = [], []
for prefix in prefixs:
    x = np.load(os.path.join(dir, prefix_to_x_filename[prefix]))
    y = np.load(os.path.join(dir, prefix_to_y_filename[prefix]))
    X.append(x)
    Y.append(y)

X = np.concatenate(X)
Y = np.concatenate(Y)
print(X.shape)
print(Y.shape)
train_size = int(X.shape[0] * 0.9)
np_to_tfrecords(X[:train_size], Y[:train_size], dir + 'train')
np_to_tfrecords(X[train_size:], Y[train_size:], dir + 'test')
