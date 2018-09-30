import numpy as np
from sigmoid import *


def sigmoid_gradient(z):
    g = np.zeros(z.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the gradient of the sigmoid function evaluated at
    #                each value of z (z can be a matrix, vector or scalar)
    #

    g = sigmoid(z) * (1 - sigmoid(z))

    # ===========================================================

    return g
