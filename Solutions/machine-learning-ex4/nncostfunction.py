import numpy as np
from sigmoid import *


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    # Useful value
    m = y.size

    # You need to return the following variables correctly
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    # ===================== Your Code Here =====================
    # Instructions : You should complete the code by working thru the
    #                following parts
    #
    # Part 1 : Feedforward the neural network and return the cost in the
    #          variable cost. After implementing Part 1, you can verify that your
    #          cost function computation is correct by running ex4.py
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         theta1_grad and theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to theta1 and theta2 in theta1_grad and
    #         theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to theta1_grad
    #               and theta2_grad from Part 2.
    #

    Y = np.zeros((m, num_labels))  # 5000 x 10

    for i in range(m):
        Y[i, y[i]-1] = 1

    a1 = np.c_[np.ones(m), X]  # 5000 x 401
    a2 = np.c_[np.ones(m), sigmoid(np.dot(a1, theta1.T))]  # 5000 x 26
    hypothesis = sigmoid(np.dot(a2, theta2.T))  # 5000 x 10

    reg_theta1 = theta1[:, 1:]  # 25 x 400
    reg_theta2 = theta2[:, 1:]  # 10 x 25

    cost = np.sum(-Y * np.log(hypothesis) - np.subtract(1, Y) * np.log(np.subtract(1, hypothesis))) / m \
            + (lmd / (2 * m)) * (np.sum(reg_theta1 * reg_theta1) + np.sum(reg_theta2 * reg_theta2))

    e3 = hypothesis - Y  # 5000 x 10
    e2 = np.dot(e3, theta2) * (a2 * np.subtract(1, a2))  # 5000 x 26
    e2 = e2[:, 1:]  # drop the intercept column  # 5000 x 25

    delta1 = np.dot(e2.T, a1)  # 25 x 401
    delta2 = np.dot(e3.T, a2)  # 10 x 26

    p1 = (lmd / m) * np.c_[np.zeros(hidden_layer_size), reg_theta1]
    p2 = (lmd / m) * np.c_[np.zeros(num_labels), reg_theta2]

    theta1_grad = p1 + (delta1 / m)
    theta2_grad = p2 + (delta2 / m)

    # ====================================================================================
    # Unroll gradients
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad
