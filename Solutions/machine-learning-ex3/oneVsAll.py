import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    # ===================== Your Code Here =====================
    # Instructions : You should complete the following code to train num_labels
    #                logistic regression classifiers with regularization
    #                parameter lambda
    #
    #
    # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
    #       whether the ground truth is true/false for this class
    #
    # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
    #       function. It is okay to use a for-loop (for c in range(num_labels) to
    #       loop over the different classes
    #

    for i in range(num_labels):
        initial_theta = np.zeros((n + 1, 1))
        iclass = i if i else 10
        y_i = np.array([1 if x == iclass else 0 for x in y])
        print('Optimizing for handwritten number {}...'.format(i))

        def cost_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[0]

        def grad_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[1]

        theta, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=initial_theta, maxiter=100, disp=False, full_output=True)
        print('Done')
        all_theta[i] = theta

    return all_theta
