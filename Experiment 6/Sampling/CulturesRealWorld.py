import numpy
import numpy as np
from preflibtools.instances import OrdinalInstance


def getReal(path):
    instance = OrdinalInstance(path)
    profile = np.array(instance.full_profile())
    profile = numpy.apply_along_axis(lambda x: x[0], 2, profile)
    profile = profile - 1
    return profile
