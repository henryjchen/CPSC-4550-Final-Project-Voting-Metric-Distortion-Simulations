from multiprocessing import Process
import Experiment

if __name__ == "__main__":
    Experiment.makeListOfSamplingModes()
    numberOfSamplingmethods = len(Experiment.listOfSamplingModes)
    processes = []
    for i in range(numberOfSamplingmethods):
        p = Process(target=Experiment.experimentParallelS, args=(i,))
        p.start()
        processes.append(p)
    for i in range(numberOfSamplingmethods):
        processes[i].join()

