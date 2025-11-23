"""
RaDiUS (beta-Random Dictatorship on the Uncovered Set) algorithm implementation.

RaDiUS first computes the uncovered set based on a beta threshold, then
samples a random voter and returns their favorite candidate from the uncovered set.
"""

import numpy as np

from .common import compute_pairwise_matrix


def _covers(a, b, s, beta, m_candidates):
    """
    Checks if candidate 'a' covers candidate 'b' in the uncovered set definition.
    
    Candidate 'a' covers 'b' if:
    1. 'a' beats 'b' by at least beta threshold (s[a, b] >= beta)
    2. Every candidate that beats 'a' (by >= beta) also beats 'b' (by >= beta)
    
    Args:
        a (int): Index of first candidate.
        b (int): Index of second candidate.
        s (np.array): Pairwise preference matrix.
        beta (float): Consensus threshold parameter.
        m_candidates (int): Total number of candidates.
        
    Returns:
        bool: True if 'a' covers 'b', False otherwise.
    """
    # Condition 1
    if s[a, b] < beta:
        return False
        
    # Condition 2
    for c in range(m_candidates):
        if c == a or c == b:
            continue
            
        if s[c, a] >= beta:
            # We found a 'c' that beats 'a'.
            # We must now check if it also beats 'b'.
            if s[c, b] < beta:
                # Condition 2 failed. 'a' does not cover 'b'.
                return False
                
    # If we get through the loop, all 'c' that beat 'a' also beat 'b'.
    return True


def run_radius(rankings, beta):
    """
    Runs the beta-Random Dictatorship on the (Weighted) Uncovered Set (RaDiUS) algorithm.
    
    Args:
        rankings (np.array): The (n_voters, m_candidates) ordinal ballots.
        beta (float): The 'consensus' parameter (e.g., 0.75)
        
    Returns:
        np.array: Probability distribution over candidates (m_candidates,).
                  Returns a degenerate distribution with probability 1.0 on the winner.
    """
    n_voters, m_candidates = rankings.shape
    
    # 1. Compute the pairwise matrix
    s = compute_pairwise_matrix(rankings)
    
    # 2. Find the uncovered set U
    uncovered_set = set(range(m_candidates))
    
    for b in range(m_candidates):
        if b not in uncovered_set:
            continue
            
        for a in range(m_candidates):
            if a == b:
                continue
            
            if _covers(a, b, s, beta, m_candidates):
                uncovered_set.discard(b)
                break
    # 3. Handle the candidate pool
    if len(uncovered_set) == 0:
        candidate_pool = list(range(m_candidates))
    else:
        candidate_pool = list(uncovered_set)

    # 4. Pick a random voter and their favorite in U
    v_index = np.random.randint(0, n_voters)
    v_ranking = rankings[v_index]
    
    winner = None
    for candidate in v_ranking:
        if candidate in candidate_pool:
            winner = candidate
            break
            
    if winner is None:
        winner = candidate_pool[0]
    
    # Return a degenerate distribution with probability 1.0 on the winner
    dist = np.zeros(m_candidates)
    dist[winner] = 1.0
    return dist

