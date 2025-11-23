import pulp as pl
from pulp import LpConstraint
import numpy as np

def profileToSuppportMatrix(profile,n,m):
    matrix = np.zeros((m,m))
    for i in range(0,n):
        for x in range(0,m-1):
            for y in range(x+1,m):
                c1 = profile[i][x]
                c2 = profile[i][y]
                matrix[c1, c2] += 1
    return matrix

"""
compute the Majority margin matrix of a given profile
"""
def profileToMajorityMarginMatrix(profile, n, m):
    matrix = np.zeros((m, m))
    for i in range(0, n):
        for x in range(0, m - 1):
            for y in range(x + 1, m):
                c1 = profile[i][x]
                c2 = profile[i][y]
                matrix[c1, c2] += 1
                matrix[c2, c1] -= 1
    return matrix


def profileToMajorityRelation(profile, n,m ):
    matrix = profileToMajorityMarginMatrix(profile,n,m)
    return np.sign(matrix)

def isTopChoice(preferences, a):
    preferences = np.array(preferences)
    tes = list(filter(lambda x: x[0] == a, preferences))
    return len(tes) > 0

def computeDistortion(profile, lottery, optimal_alt):
    numVot, numAlt = np.shape(profile)
    model = pl.LpProblem("Distortion", pl.LpMaximize)
    arrAlt = np.arange(numAlt)
    arrVot = np.arange(numVot)
    arrAlt = list(map(lambda x: "a" + str(x), arrAlt))
    arrVot = list(map(lambda x: "v" + str(x), arrVot))
    x = pl.LpVariable.dicts("x", (alt for alt in arrAlt), lowBound = 0, cat = "Continuous")
    d = pl.LpVariable.dicts("d", ((alt, voter) for alt in arrAlt for voter in arrVot), lowBound=0, cat="Continuous")

    model += LpConstraint(x[arrAlt[optimal_alt]], rhs=0, sense=pl.LpConstraintEQ, name="opt_alt")
    model += LpConstraint(pl.lpSum(d[arrAlt[optimal_alt],voter] for voter in arrVot), rhs=1, sense=pl.LpConstraintEQ, name="normalization")
    model += pl.lpSum(d[arrAlt[i], voter] * lottery[i] for voter in arrVot for i in range(numAlt)) #Objective

    for v in range(0, numVot):
        for i in range(0, numAlt):
            for j in range(i, numAlt):
                model += LpConstraint(x[arrAlt[profile[v][i]]] - x[arrAlt[profile[v][j]]] - 2*d[arrAlt[optimal_alt], arrVot[v]],
                                  rhs=0, sense=pl.LpConstraintLE, name="T1_"+str(v)+":"+str(profile[v][i])+">"+str(profile[v][j]))

    for v in range(0, numVot):
        for i in range(0, numAlt):
            for j in range(i, numAlt):
                model+= LpConstraint(d[arrAlt[profile[v][i]], arrVot[v]]-d[arrAlt[optimal_alt], arrVot[v]] - x[arrAlt[profile[v][j]]],
                                 rhs=0, sense=pl.LpConstraintLE, name="T2_"+str(v)+":"+str(profile[v][i])+">"+str(profile[v][j]))

    for v in range(0, numVot):
        for i in range(0,numAlt):
            model+= LpConstraint(d[arrAlt[optimal_alt], arrVot[v]]+d[arrAlt[i], arrVot[v]] - x[arrAlt[i]],
                                 rhs=0, sense=pl.LpConstraintGE, name="T3_"+str(v)+":"+str(profile[v][i]))

    pl.GUROBI(msg=False).solve(model)
    return model.objective.value()


def get_worst_case_distortion(profile, distribution):
    """
    Compute worst-case distortion by assuming adversarial nature.
    
    This function implements the "Nature is adversarial" assumption:
    We don't know who the true optimal candidate is, so we assume
    the worst-case scenario where the ground truth metric space
    makes our algorithm look worst.
    
    Parameters:
    -----------
    profile : numpy.ndarray, shape (n_voters, m_candidates)
        Preference profile
    distribution : list or numpy.ndarray, length m_candidates
        Probability distribution (lottery) over candidates
        
    Returns:
    --------
    float : Maximum distortion across all possible optimal candidates
    """
    numVot, numAlt = np.shape(profile)
    
    # Iterate over all candidates, assuming each is optimal
    max_distortion = -np.inf
    
    for c in range(numAlt):
        # Assume candidate c is the true optimal
        distortion = computeDistortion(profile, distribution, optimal_alt=c)
        
        # Track the maximum (worst-case)
        if distortion is not None and distortion > max_distortion:
            max_distortion = distortion
    
    return max_distortion