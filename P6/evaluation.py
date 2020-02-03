import numpy as np
from scipy.spatial.distance import pdist, squareform


def correlation(X, predicted):
    proximity = squareform(pdist(X))
    incidence = np.zeros(shape=(predicted.shape[0], predicted.shape[0]))

    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[0]):
            if i != j and predicted[i] == predicted[j]:
                incidence[i, j] = 1

    return np.corrcoef(proximity.flatten(), incidence.flatten())[0, 1]