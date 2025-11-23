import Experiment
import os
from Sampling import CulturesRealWorld
import Helper
import time
import numpy as np
import re

### These experiments rely on the Spotify Daily dataset, which is availble at PrefLib: https://preflib.github.io/PrefLib-Jekyll/dataset/00047
### For executing the code, you need to download the spotifydaily dataset and need to move it in the same directory as this document.
### Also, please makre sure that the name of the directory matches the string used in the code "spotifyday"

numbers = re.compile(r'(\d+)')


def helperSpotify(dataFile):
    arr = []
    dataF = dataFile
    for file in os.listdir(dataF):
        if not os.path.isdir(dataF + "/" + file):
            if file.endswith(".soc"):
                arr.append(os.path.join(dataF + "/" + file))
    return arr


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def spotifyDaily():
    listi = helperSpotify("./spotifyday")
    listi = sorted(listi, key=numericalSort)
    location = "./Data/Spotify.txt"
    start_time = time.time()
    f = open(location, "w")
    t = -1
    for path in listi:
        t = t + 1
        profile = CulturesRealWorld.getReal(path)
        dic = {}
        n, m = np.shape(profile)
        dic["profile"] = profile.tolist()
        dic["t"] = t
        dic["m"] = m
        dic["n"] = n
        dic["RealWorldData"] = str(path)
        start = time.time()
        for e in Experiment.listOfRandFunctions:
            lottery = e[1](profile)
            dic[e[0] + "Lottery"] = lottery
            dic[e[0]] = max(Helper.computeDistortion(profile, lottery, alt) for alt in range(m))
        dic["time"] = time.time() - start
        data = dic
        f.write(str(data))
        f.write("\n")
        print(time.time() - start_time)
    f.close()


if __name__ == "__main__":
    spotifyDaily()


