import numpy as np

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #

    Xt = np.transpose(X)
    theta = np.linalg.pinv(Xt.dot(X)).dot(Xt).dot(y)

    return theta
