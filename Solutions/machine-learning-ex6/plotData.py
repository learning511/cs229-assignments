import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.scatter(X[pos, 0], X[pos, 1], marker="+", c='b')
    plt.scatter(X[neg, 0], X[neg, 1], marker="o", c='y', s=15)
