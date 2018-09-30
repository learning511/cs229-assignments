import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    hypothesis = sigmoid(np.dot(X, theta))

    reg_theta = theta[1:]

    cost = np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)) / m \
           + (lmd / (2 * m)) * np.sum(reg_theta * reg_theta)

    normal_grad = (np.dot(X.T, hypothesis - y) / m).flatten()

    grad[0] = normal_grad[0]
    grad[1:] = normal_grad[1:] + reg_theta * (lmd / m)

    # ===========================================================

    return cost, grad
