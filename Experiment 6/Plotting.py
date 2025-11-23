import matplotlib.pyplot as plt
import ReadData
import numpy as np

markersAlt = [".", ",", "o", "v", "^", "<", ">"]
markers = ["o", "o", "o", "o", "o", "o", "o"]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


def plotFancy(data, vec, func, nrange):
    for i in range(len(vec)):  # for each of the up to 7 features
        mi = markers[i]  # marker for ith feature
        xi = nrange  # x array for ith feature .. here is where you would generalize      different x for every feature
        yi = ReadData.selectSpecificAll(data, vec[i], func, nrange)  # y array for ith feature
        ci = colors[i]  # color for ith feature
        plt.xticks(nrange)
        plt.plot(xi, yi, marker=mi, color=ci, markersize=4)
    plt.show()


data = ReadData.extractdata2("Data/SampledData/Candidates10/Sampling_EuclideanCube1.txt")
vec = ["RandomDictatorship", "CRWW", "C2ML", "C1ML", "PluralityVeto"]
nrange = [11 + 10 * x for x in range(0, 5)]
plotFancy(data, vec, np.average, nrange)