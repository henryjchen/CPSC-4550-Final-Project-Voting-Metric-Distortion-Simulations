import numpy as np

def computeRD(profile):
    n, m = np.shape(profile)
    preferences = np.array(profile)
    prob = [0] * m
    for i in range(m):
        tes = list(filter(lambda x: x[0] == i, preferences))
        prob[i] = len(tes) / n
    return prob
