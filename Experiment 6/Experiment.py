import time
import random as rd
import numpy as np
from functools import partial
import Helper
from Sampling import CulturesPrefSampling
from RSCFs import PluralityVeto
from RSCFs import C1ML
from RSCFs import C2ML
from RSCFs import CRWW
from RSCFs import RD


mrange = [5, 10, 15]
nrange = [11 + 10 * x for x in range(0, 20)]
num_iterations = 1000


listOfRandFunctions = [["PluralityVeto", PluralityVeto.computePluralityVeto], ["RandomDictatorship", RD.computeRD], ["C1ML", C1ML.computeC1ML], ["C2ML", C2ML.computeC2ML], ["CRWW",CRWW.compute_CCRW]]
listOfSamplingModes = []
rng = np.random.default_rng(seed=42)


"""
creates a list of the considered disributions
"""
def makeListOfSamplingModes():
    string = "ImpartialCulture"
    model = string
    fun = partial(CulturesPrefSampling.impartial)
    listOfSamplingModes.append([model, fun])
    string = "EuclideanBall"
    for i in CulturesPrefSampling.defaultBallValues():
        model = string + str(i)
        fun = partial(CulturesPrefSampling.euclideanBall, numDim=i)
        listOfSamplingModes.append([model, fun])
    string = "EuclideanCube"
    for i in CulturesPrefSampling.defaultCubeValues():
        model = string + str(i)
        fun = partial(CulturesPrefSampling.euclideanCube, numDim=i)
        listOfSamplingModes.append([model, fun])
    string = "Mallows"
    for i in CulturesPrefSampling.getDefaultMallowsValues():
        model = string + str(i)
        fun = partial(CulturesPrefSampling.mallowsNormalized, phi=i)
        listOfSamplingModes.append([model, fun])
    string = "UrnModel"
    for i in CulturesPrefSampling.getDefaultUrnModelsValues():
        model = string + str(i)
        fun = partial(CulturesPrefSampling.urn, alpha=i)
        listOfSamplingModes.append([model, fun])


def experimentParallel(samplingMethod, m):
    makeListOfSamplingModes()
    outfile = "./Data/SampledData"
    sampling = listOfSamplingModes[samplingMethod]
    path = outfile + "/" + "Candidates" + str(m) + "/Sampling_" + sampling[0] + ".txt"
    f = open(path, "w")
    for n in nrange:
        start_time = time.time()
        data = singlerun(m, n, sampling)
        for i in range(len(data)):
            f.write(str(data[i]))
            f.write("\n")
        print(time.time() - start_time)
    f.close()

def experimentParallelS(samplingMethod):
    for m in mrange:
        experimentParallel(samplingMethod,m)

def singlerun(m, n, sampling):
    rd.seed(90807)
    np.random.seed(90807)
    data = []
    for iteration in range(0, num_iterations):
        seed = rng.integers(0, 4294967295)
        dic = {}
        dic["samplingMethod"] = sampling[0]
        profile = sampling[1](n, m, seed=seed)
        profile = np.array(profile)
        dic["profile"] = profile.tolist()
        dic["m"] = m
        dic["n"] = n
        dic["seed"] = int(seed)
        start = time.time()
        for e in listOfRandFunctions:
            lottery = e[1](profile)
            dic[e[0] + "Lottery"] = lottery
            dic[e[0]] = max(Helper.computeDistortion(profile, lottery, alt) for alt in range(m))
        dic["time"] = time.time() - start
        data.append(dic)
    return data

