import numpy as np

def get_distance_matrix(voter_coords, candidate_coords):
    """
    Calculates the Euclidean distance matrix between all voters
    and all candidates.
    
    Args:
        voter_coords (np.array): (n, 2) array of voter coordinates.
        candidate_coords (np.array): (m, 2) array of candidate coordinates.
        
    Returns:
        np.array: (n, m) distance matrix.
    """
    # (n, 1, 2) - (1, m, 2) -> (n, m, 2)
    diffs = voter_coords[:, np.newaxis, :] - candidate_coords[np.newaxis, :, :]
    
    # (n, m, 2) -> (n, m)
    dist_matrix = np.linalg.norm(diffs, axis=2)
    
    return dist_matrix

def get_social_costs(dist_matrix):
    """
    Calculates the social cost (average distance) for each candidate.
    
    Args:
        dist_matrix (np.array): (n, m) distance matrix.
        
    Returns:
        np.array: (m,) array of social costs.
    """
    # Mean across the voter axis (axis 0)
    return np.mean(dist_matrix, axis=0)

def get_true_optimal_candidate(social_costs):
    """
    Finds the index of the candidate with the minimum social cost.
    
    Args:
        social_costs (np.array): (m,) array of social costs.
        
    Returns:
        int: The index of the optimal candidate.
    """
    return np.argmin(social_costs)
