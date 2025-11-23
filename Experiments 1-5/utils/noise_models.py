import numpy as np

def _sample_plackett_luce(scores):
    """
    Samples a single ranking from a Plackett-Luce distribution.
    
    Args:
        scores (np.array): A 1D array of "skill" scores (w_i).
        
    Returns:
        np.array: A 1D array (a ranking) of candidate indices.
    """
    m_candidates = len(scores)
    ranking = []
    
    # Create a pool of candidates to draw from
    # We use their indices (0, 1, ..., m-1)
    pool = list(range(m_candidates))
    current_scores = np.array(scores) # Copy scores
    
    for _ in range(m_candidates):
        # 1. Calculate probabilities
        # Prevent division by zero if all scores are zero
        total_score = np.sum(current_scores)
        if total_score <= 1e-9:
            # If all remaining scores are 0, pick a remaining
            # candidate uniformly at random.
            probs = np.ones(len(current_scores)) / len(current_scores)
        else:
            probs = current_scores / total_score
        
        # 2. Sample a winner from the pool
        # np.random.choice requires an index into the current pool
        winner_pool_idx = np.random.choice(len(pool), p=probs)
        
        # 3. Get the *original* candidate index
        winner_candidate_idx = pool.pop(winner_pool_idx)
        
        # 4. Add to ranking
        ranking.append(winner_candidate_idx)
        
        # 5. Remove the winner's score from the distribution
        # (by setting it to 0 and removing from the pool array)
        current_scores = np.delete(current_scores, winner_pool_idx)

    return np.array(ranking)

def generate_noisy_rankings(dist_matrix, temperature):
    """
    Generates ordinal ballots for all voters by applying a
    Plackett-Luce noise model to the ground-truth distances.
    
    Args:
        dist_matrix (np.array): The (n, m) ground-truth distance matrix.
        temperature (float): The "variance" or "noise" parameter (tau).
                             - Low temp (e.g., 0.01): Perfect, rational rankings.
                             - High temp (e.g., 10.0): Random, noisy rankings.
                             
    Returns:
        np.array: The (n, m) array of "noisy" ordinal ballots.
    """
    n_voters, m_candidates = dist_matrix.shape
    
    # 1. Convert distances (cost) to utilities (higher is better)
    # We add a small epsilon to avoid division by zero for d=0
    utilities = 1.0 / (dist_matrix + 0.01)
    
    # 2. Calculate Plackett-Luce scores using the temperature
    # w = exp(utility / temperature)
    # Clamp temperature to avoid div-by-zero
    safe_temp = max(temperature, 1e-9)
    
    # Calculate scores before exponentiating
    raw_scores = utilities / safe_temp

    max_scores_per_voter = np.max(raw_scores, axis=1, keepdims=True)
    stabilized_scores = raw_scores - max_scores_per_voter
    
    # This line is now numerically safe and will not overflow
    pl_scores = np.exp(stabilized_scores)
    
    # 3. Generate a ranking for each voter
    ordinal_ballots = np.zeros((n_voters, m_candidates), dtype=int)
    
    for i in range(n_voters):
        voter_scores = pl_scores[i]
        ordinal_ballots[i] = _sample_plackett_luce(voter_scores)
        
    return ordinal_ballots