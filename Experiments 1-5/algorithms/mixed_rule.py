"""
Mixed Rule algorithm implementation.

The Mixed Rule combines Maximal Lotteries and RaDiUS by randomly selecting
one of the two algorithms according to a theoretically optimal probability distribution.
This achieves better distortion bounds than either algorithm alone.
"""

import numpy as np

from .ml import run_maximal_lotteries
from .radius import run_radius


def _get_mixed_rule_params(B=0.876353):
    """
    Computes the mixing parameters for the Mixed Rule algorithm.
    
    Calculates the probability p_ml of running Maximal Lotteries and provides
    a function to sample beta values for RaDiUS according to the theoretical distribution.
    
    Args:
        B (float): The optimal upper bound for beta, from paper. Default: 0.876353.
        
    Returns:
        tuple: (p_ml, sample_beta) where:
            - p_ml (float): Probability of running ML. Should be approx. 0.55.
            - sample_beta (callable): Function that returns a sampled beta value.
    """
    # 1. Calculate 'p' (probability of running ML, following integral equation from 
    # Charikar et al. (2024) and Frank and Lederer (2025))
    def integral_part(beta):
        return 0.5 * (np.log(1 + beta) - np.log(1 - beta))
    
    integral_val = integral_part(B) - integral_part(0.5)
    p_ml = 1.0 / (1.0 + integral_val)
    
    # 2. Define the sampling function for beta
    integral_part_half = integral_part(0.5)
    
    def sample_beta():
        u = np.random.rand() 
        C = 2 * (u * integral_val + integral_part_half)
        exp_C = np.exp(C)
        beta = (exp_C - 1) / (exp_C + 1)
        return beta
        
    return p_ml, sample_beta


def run_mixed_rule(rankings, B=0.876353, p_ml_override=None):
    """
    Runs the final barrier-breaking "ML Mixed with RaDiUS" / "CRWW" algorithm
    
    Args:
        rankings (np.array): The (n_voters, m_candidates) ordinal ballots.
        B (float): The optimal upper bound for beta, from paper [cite: 642].
        p_ml_override (float, optional): Override the theoretical p_ml value.
                                        If None, uses the theoretical value from B.
        
    Returns:
        np.array: Probability distribution over candidates (m_candidates,).
                  Returns the distribution from either ML or RaDiUS, depending on
                  the mixing probability.
    """
    # 1. Get the mixing parameters
    if p_ml_override is not None:
        p_ml = p_ml_override
        _, sample_beta = _get_mixed_rule_params(B)
    else:
        p_ml, sample_beta = _get_mixed_rule_params(B)
    
    # 2. With probability p, run ML. Otherwise, run RaDiUS
    if np.random.rand() < p_ml:
        return run_maximal_lotteries(rankings)
    else:
        beta = sample_beta()
        return run_radius(rankings, beta=beta)

