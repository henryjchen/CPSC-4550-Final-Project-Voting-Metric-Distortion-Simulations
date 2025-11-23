import numpy as np
import Helper
import itertools
import random as rd
from RSCFs import C2ML

B=0.876353
#The primitive of 1/(1-x^2) is arctanh. Hence, the integral from 0.5 to B over 1/(1-x^2) arctanh(B)-arctanh(0.5)
p=1/(1+np.arctanh(B)-np.arctanh(0.5))


def rho_cumulative(x,y):
    return p/(1-p)*(np.arctanh(y)-np.arctanh(x))


def compute_weighted_uncovered_set(profile, m, n, treshold):
    matrix = Helper.profileToSuppportMatrix(profile, n, m)
    matrix = (matrix >= treshold)
    covered=set()
    for x,y in itertools.permutations(np.arange(m),2):
        if y not in covered and matrix[x,y] and all([not matrix[z,x] or matrix[z,y] for z in range(m)]):
            covered.add(y)
    return set(np.arange(m)).difference(covered)


def find_best_element(preference, set):
    for ele in preference:
        if ele in set:
            return ele


def compute_RaDiUS(profile,m,n,treshold):
    uncovered_set=compute_weighted_uncovered_set(profile,m,n,treshold)
    reduced_profile=[find_best_element(preference, uncovered_set) for preference in profile]
    lottery = np.zeros(m, dtype=float)
    for i in uncovered_set:
        numb = reduced_profile.count(i)
        lottery[i]=numb/n
    return lottery


#for computing the integral over the radius function, we note that for every beta in (t/n, (t+1)/n] for t\in {(n+1)/2,dots,n-1},
#the weighted uncovered set is fixed, so the beta function returns always the same out put. We thus integrate by computing
#the integral of each of these subparts individually.
def compute_RaDiUS_integral(profile,m,n):
    radius_integral = np.zeros(m, dtype=float)
    # Special case: first iteration starts at 0.5
    radius = compute_RaDiUS(profile, m, n, int((n + 1) / 2))
    integral = rho_cumulative(0.5, (n + 1) / (2 * n))
    for x in range(m):
        radius_integral[x] += radius[x] * integral
    # Standard case: integrate from
    for treshold in range(int((n + 1) / 2), int(n * B)):
        radius = compute_RaDiUS(profile, m, n, treshold + 1)
        integral = rho_cumulative(treshold / n, (treshold + 1) / n)
        for x in range(m):
            radius_integral[x] += radius[x] * integral
    # Special case: last iteration goes to B
    radius = compute_RaDiUS(profile, m, n, (1 + int(n * B)) / n)
    integral = rho_cumulative(int(n * B) / n, B)
    for x in range(m):
        radius_integral[x] += radius[x] * integral
    return radius_integral


def compute_CCRW(profile):
    n, m = np.shape(profile)
    ml = C2ML.computeC2ML(profile)
    radius = compute_RaDiUS_integral(profile, m, n)
    #return p*np.array(ml) + (1-p)*np.array(radius)
    value = p*np.array(ml) + (1-p)*np.array(radius)
    value = value / np.sum(value)
    value = value.tolist()
    return value


#Data radius implementation by approximating it by sampling according to rho
def test_radius_implementation(iterations, profile, m, n):
    rule_output = compute_RaDiUS_integral(profile, m, n)
    lottery = np.zeros(m, dtype=float)
    for i in range(iterations):
        x = rd.uniform(np.arctanh(0.5) * p / (1 - p), np.arctanh(B) * p / (1 - p))
        beta = np.tanh(x * (1 - p) / p)  # Inverse of the CDF of rho
        lottery += compute_RaDiUS(profile, m, n, beta * n)
    lottery=lottery/iterations
    print(lottery - rule_output)

