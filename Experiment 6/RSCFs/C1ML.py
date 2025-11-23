import numpy as np
import pulp as pl
import Helper

def computeC1ML(profile):
    n, m = np.shape(profile)
    matrix = Helper.profileToMajorityRelation(profile,n,m)
    model = pl.LpProblem("C2ML", pl.LpMaximize)
    lottery = pl.LpVariable.dict("lottery", (i for i in range(m)), lowBound=0,upBound=1, cat = "Continuous")
    for i in range(m):
        model += pl.lpSum([matrix[j,i] * lottery[j] for j in range(m)]) >= 0
    model += pl.lpSum([lottery[j] for j in range(m)]) == 1
    pl.GUROBI(msg=False).solve(model)
    probs = [0] * m
    for i in range(m):
        probs[i] = lottery[i].varValue
    return probs

