import Helper
import numpy as np
import networkx as nx

"""
Sets up matching Graph to check whether a is a PluralityVeto winner in the profile
"""
def preferencesProfileToGraph(preferences, a):
    preferences = np.array(preferences)
    numVot, numAlt = np.shape(preferences)
    topChoice = list(map(lambda x: x[0], preferences))
    graph = nx.Graph()
    graph.add_nodes_from(list(range(numVot)), bipartite=0)
    graph.add_nodes_from(list(range(numVot, 2 * numVot)), bipartite=1)
    for i in range(numVot):
        for k in range(numVot):
            first = np.where(preferences[k, :] == a)
            second = np.where(preferences[k, :] == topChoice[i])
            if (first <= second):
                graph.add_edge(i, k + numVot)
    return graph

"""
checking if a candidate is a pluralityVeto winner
"""
def isPluVeto(profile, a):
    prof = np.array(profile)
    numVot, numAlt = np.shape(prof)
    graph = preferencesProfileToGraph(prof, a)
    return len(nx.max_weight_matching(graph)) == numVot and Helper.isTopChoice(profile, a)

"""
computes the randomized Plurality-Veto rule
"""
def computePluralityVeto(profile):
    n, m = np.shape(profile)
    pluralityVetoSet = []
    for i in range(m):
        if isPluVeto(profile, i):
            pluralityVetoSet.append(i)
    lottery = [0] * m
    for i in pluralityVetoSet:
        lottery[i] = 1.0 / len(pluralityVetoSet)
    return lottery
