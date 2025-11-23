import numpy as np

def generate_instance(n_voters, m_candidates):
    """
    Generates a random instance with voters and candidates in a 2D space.
    
    Args:
        n_voters (int): Number of voters.
        m_candidates (int): Number of candidates.
        
    Returns:
        tuple: (voter_coords, candidate_coords, None, None) where:
            - voter_coords (np.array): (n_voters, 2) array of voter coordinates in [0,1]².
            - candidate_coords (np.array): (m_candidates, 2) array of candidate coordinates in [0,1]².
            - None: Placeholder for precomputed distance matrix (not used).
            - None: Placeholder for precomputed rankings (not used).
    """
    voter_coords = np.random.rand(n_voters, 2)
    candidate_coords = np.random.rand(m_candidates, 2)
    
    return voter_coords, candidate_coords, None, None


def generate_rd_trap_instance(n_voters=100, m_candidates=5):
    """
    Generates a classic outlier N-1 vs. 1 trap instance for the RaDiUS algorithm. 
    Places N-1 voters at one location near an optimal candidate, and 1 outlier 
    voter at a different location near a suboptimal candidate.
    
    Args:
        n_voters (int): Number of voters.
        m_candidates (int): Number of candidates.
        
    Returns:
        tuple: (voter_coords, candidate_coords, None, None) where:
            - voter_coords (np.array): (n_voters, 2) array of voter coordinates.
            - candidate_coords (np.array): (m_candidates, 2) array of candidate coordinates.
            - None: Placeholder for precomputed distance matrix (not used).
            - None: Placeholder for precomputed rankings (not used).
    """
    if m_candidates < 2:
        return np.random.rand(n_voters, 2), np.random.rand(m_candidates, 2), None, None
        
    voter_coords = np.zeros((n_voters, 2))
    candidate_coords = np.zeros((m_candidates, 2))
    
    candidate_coords[0] = [0.5, 0.5]
    candidate_coords[1] = [1.5, 0.5]
    candidate_coords[2:] = [100.0, 100.0] 
        
    n_group_A = n_voters - 1 
    voter_coords[:n_group_A] = [0.5, 0.5]
    voter_coords[n_group_A:] = [1.5, 0.5]
    
    return voter_coords, candidate_coords, None, None


def generate_ml_trap_instance(n_voters=100, m_candidates=5):
    """
    Generates a "Majority Tyranny" trap instance designed to fail Maximal Lotteries (ML).
    
    Args:
        n_voters (int): Number of voters (default: 100)
        m_candidates (int): Number of candidates (default: 5)
    
    Returns:
        tuple: (voter_coords, candidate_coords, None, None)
            - voter_coords: (n_voters, 2) array of voter coordinates
            - candidate_coords: (m_candidates, 2) array of candidate coordinates
    """
    if m_candidates < 2:
        return np.random.rand(n_voters, 2), np.random.rand(m_candidates, 2), None, None
    
    # Initialize coordinate arrays
    voter_coords = np.zeros((n_voters, 2), dtype=float)
    candidate_coords = np.zeros((m_candidates, 2), dtype=float)
    
    # Split voters: 51% majority vs 49% minority
    n_group_1 = n_voters // 2 + 1  # The "Wrong" Majority (51%)
    n_group_2 = n_voters - n_group_1  # The "Right" Minority (49%)
    
    # Voter Group 1 (The "Wrong" Majority): Place at [0.49, 0.5]
    voter_coords[:n_group_1] = [0.49, 0.5]
    
    # Voter Group 2 (The "Right" Minority): Place at [1.0, 0.5]
    voter_coords[n_group_1:] = [1.0, 0.5]
    
    # Candidate 0: Place at [0.0, 0.5]
    candidate_coords[0] = [0.0, 0.5]
    
    # Candidate 1: Place at [1.0, 0.5]
    candidate_coords[1] = [1.0, 0.5]
    
    # Dummy Candidates: Place far away (between [2.0, 2.0] and [3.0, 3.0])
    if m_candidates > 2:
        for i in range(2, m_candidates):
            candidate_coords[i] = np.random.uniform([2.0, 2.0], [3.0, 3.0])
    
    return voter_coords, candidate_coords, None, None
