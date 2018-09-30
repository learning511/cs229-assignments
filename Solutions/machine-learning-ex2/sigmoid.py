import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)

    # ===================== Your Code Here =====================
    # Instructions : Compute the sigmoid of each value of z (z can be a matrix,
    #                vector or scalar
    #
    # Hint : Do not import math

    g = 1 / (1 + np.exp(-z))

    return g
