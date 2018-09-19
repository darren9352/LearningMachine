from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def get_mnist_idx(y_test, want_class) :
    for i in range(10000) :
        if np.argmax(y_test[i]) == want_class :
            return i

def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    This is adapted from the Keras function with the same name.
    :param y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
    :param num_classes: num_classes: total number of classes.
    :return: A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def get_mnist_data(datadir='/tmp/'):

    import mnist

    X_test = mnist.test_images() / 255.
    Y_test = mnist.test_labels()

    X_test = np.expand_dims(X_test, -1)

    X_test = X_test[0:1666]
    Y_test = Y_test[0:1666]

    Y_test = to_categorical(Y_test, num_classes=10)
    return X_test, Y_test
