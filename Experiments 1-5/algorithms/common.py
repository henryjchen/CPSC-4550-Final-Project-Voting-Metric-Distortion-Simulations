"""
Common utilities shared across voting algorithms.

This module contains helper functions used by multiple voting algorithms,
such as pairwise preference computation.
"""

import numpy as np


def _voter_prefers(ranking, a, b):
    """
    Checks if a voter prefers candidate 'a' over candidate 'b' based on their ranking.
    
    Args:
        ranking (np.array): A single voter's ranking of candidates.
        a (int): Index of first candidate.
        b (int): Index of second candidate.
        
    Returns:
        bool: True if candidate 'a' is preferred over 'b' (ranked higher), False otherwise.
    """
    # Get the position (rank) of candidates 'a' and 'b' in the list
    pos_a = np.where(ranking == a)[0][0]
    pos_b = np.where(ranking == b)[0][0]
    
    # Lower position (index) means more preferred
    return pos_a < pos_b


def compute_pairwise_matrix(rankings):
    """
    Computes the pairwise preference matrix 's', where s[a, b] is the
    proportion of voters who prefer candidate 'a' over candidate 'b'.
    
    Args:
        rankings (np.array): The (n_voters, m_candidates) ordinal ballots.
        
    Returns:
        np.array: Pairwise preference matrix of shape (m_candidates, m_candidates).
                  Entry s[a, b] is the proportion of voters preferring candidate 'a' over 'b'.
    """
    n_voters, m_candidates = rankings.shape
    pairwise_matrix = np.zeros((m_candidates, m_candidates))
    
    for a in range(m_candidates):
        for b in range(m_candidates):
            if a == b:
                pairwise_matrix[a, b] = 0.5
                continue
                
            preference_count = 0
            for voter_ranking in rankings:
                if _voter_prefers(voter_ranking, a, b):
                    preference_count += 1
            
            # Store the proportion of voters who prefer candidate 'a' over 'b'
            pairwise_matrix[a, b] = preference_count / n_voters
            
    # Apply consistent rounding 
    return np.round(pairwise_matrix, decimals=5)

