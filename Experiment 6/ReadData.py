import numpy as np
import json


def extractdata2(filename):
    file = open(filename, "r")
    output = []
    line = file.readline()
    while len(line) > 0:
        line = line.replace("\'", "\"")
        data = json.loads(line)
        output.append(data)
        line = file.readline()
    output = np.array(output)
    return output


def selectSpecificAll(data, type, func, nrange):
    arr = [0]*len(nrange)
    for i in range(len(nrange)):
        arr[i] = func(selectSpecific(data,type,nrange[i]))
    return arr


def selectSpecific(data, type, numberOfVoters):
    filFunc = lambda x: x["n"] == numberOfVoters
    func = lambda x: x[type]
    filteredData = (list(filter(filFunc, data)))
    filteredData = np.array(filteredData)
    vec = np.vectorize(func)
    return (vec(filteredData))
