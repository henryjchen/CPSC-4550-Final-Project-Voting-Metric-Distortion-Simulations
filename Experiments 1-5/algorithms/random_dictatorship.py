"""
Random Dictatorship algorithm implementation.

Random Dictatorship selects a random voter and returns their top choice.
This is a simple baseline algorithm with no strategic computation.
"""

import numpy as np


def run_random_dictatorship(rankings):
    """
    Runs the Random Dictatorship algorithm.
    
    Args:
        rankings (np.array): The (n_voters, m_candidates) ordinal ballots.
        
    Returns:
        np.array: Probability distribution over candidates (m_candidates,).
                  Returns a degenerate distribution with probability 1.0 on the
                  top choice of a randomly selected voter.
    """
    n_voters, m_candidates = rankings.shape
    
    # 1. Pick a random voter
    v_index = np.random.randint(0, n_voters)
    
    # 2. Get that voter's top choice
    winner = rankings[v_index, 0]
    
    # Return a degenerate distribution with probability 1.0 on the winner
    dist = np.zeros(m_candidates)
    dist[winner] = 1.0
    return dist

