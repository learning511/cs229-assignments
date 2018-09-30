import numpy as np


def find_closest_centroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    m = X.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Go over every example, find its closest centroid, and store
    #                the index inside idx at the appropriate location.
    #                Concretely, idx[i] should contain the index of the centroid
    #                closest to example i. Hence, it should be a value in the
    #                range 0..k
    #

    means = np.zeros((m, K))

    for i in range(m):
        x = X[i]
        diff = x - centroids

        for k in range(K):
            means[i, k] = np.linalg.norm(diff[k])

    idx = np.argmin(means, axis=1)

    # ==========================================================

    return idx
