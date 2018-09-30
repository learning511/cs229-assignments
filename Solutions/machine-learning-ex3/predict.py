import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    #
    # Hint : the max function blah blah
    #

    x = np.c_[np.ones(m), x]
    h1 = sigmoid(np.dot(x, theta1.T))
    h1 = np.c_[np.ones(h1.shape[0]), h1]
    h2 = sigmoid(np.dot(h1, theta2.T))
    p = np.argmax(h2, axis=1) + 1

    return p


