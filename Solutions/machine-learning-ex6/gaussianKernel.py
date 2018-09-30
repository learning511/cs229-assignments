import numpy as np


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()

    sim = 0

    # ===================== Your Code Here =====================
    # Instructions : Fill in this function to return the similarity between x1
    #                and x2 computed using a Gaussian kernel with bandwith sigma
    #

    sim = np.exp(np.sum((x1 - x2) ** 2) / (-2*sigma**2))

    # ==========================================================

    return sim

