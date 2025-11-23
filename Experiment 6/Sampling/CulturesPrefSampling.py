import prefsampling as ps
import numpy as np

def defaultCubeValues():
    return [1,3,5,15]
def euclideanCube(numVot, numCand, numDim, seed):
    return ps.ordinal.euclidean(seed=seed, num_voters=numVot, num_candidates=numCand, num_dimensions=numDim, voters_positions=ps.EuclideanSpace.UNIFORM_CUBE, candidates_positions=ps.EuclideanSpace.UNIFORM_CUBE)


def defaultBallValues():
    return [3,5,15]
def euclideanBall(numVot, numCand, numDim, seed):
    return ps.ordinal.euclidean(num_voters=numVot, num_candidates=numCand, num_dimensions=numDim,
                                voters_positions=ps.EuclideanSpace.UNIFORM_BALL,
                                candidates_positions=ps.EuclideanSpace.UNIFORM_BALL, seed=seed)


def impartial(numVot, numCand, seed):
    return ps.ordinal.impartial(num_voters=numVot, num_candidates=numCand, seed=seed)


def getDefaultMallowsValues():
    return [0.5,0.7,0.9]
def mallowsNormalized(numVot, numCand, phi, seed):
    return ps.ordinal.norm_mallows(num_voters=numVot, num_candidates=numCand, norm_phi=phi, seed=seed)


def getDefaultUrnModelsValues():
    return [0.05, 0.15, 0.3]
def urn(numVot, numCand, alpha, seed):
    return ps.ordinal.urn(num_voters=numVot, num_candidates=numCand, alpha=alpha, seed=seed)


def generate_ml_trap_instance(n_voters=100, m_candidates=5):
    """
    Generate a "Split Electorate" trap instance designed to fail Maximal Lotteries (ML).
    
    This function creates a geometric setup that forces a perfect pairwise tie (50% vs 50%)
    between an Optimal Candidate and a Trap Candidate, such that ML is mathematically
    forced to assign them equal probability (0.5 / 0.5), despite the Trap having
    approximately 3x the social cost.
    
    Theoretical Explanation:
    -----------------------
    Voter Group 1 (The Trap Sympathizers):
    - Place exactly n_voters // 2 voters at [0.49, 0.5]
    - Distance to Trap (candidate 0 at [0.0, 0.5]) = 0.49
    - Distance to Optimal (candidate 1 at [1.0, 0.5]) = 0.51
    - They prefer Trap over Optimal
    
    Voter Group 2 (The Rational Block):
    - Place remaining voters at [1.0, 0.5]
    - Distance to Trap = 1.0
    - Distance to Optimal = 0.0
    - They prefer Optimal over Trap
    
    Pairwise Result:
    - Exactly 50% of voters prefer Trap
    - Exactly 50% of voters prefer Optimal
    - This creates a perfect tie in the majority relation
    
    ML Failure:
    - The LP solver will find a Nash Equilibrium of [0.5, 0.5] for Trap and Optimal
    - ML cannot break the tie and must assign equal probability
    
    Cost Imbalance:
    - Cost(Optimal) ≈ 0.5 × 0.51 + 0.5 × 0.0 ≈ 0.255
    - Cost(Trap) ≈ 0.5 × 0.49 + 0.5 × 1.0 ≈ 0.745
    - Distortion: 0.745 / 0.255 ≈ 2.92
    
    This proves ML fails to identify the efficient winner in symmetric ties.
    
    Parameters:
    -----------
    n_voters : int, default=100
        Total number of voters
    m_candidates : int, default=5
        Total number of candidates (must be >= 2)
    
    Returns:
    --------
    voter_coords : numpy.ndarray, shape (n_voters, 2)
        Coordinates of all voters in 2D space
    candidate_coords : numpy.ndarray, shape (m_candidates, 2)
        Coordinates of all candidates in 2D space
    optimal : None
        Placeholder for optimal candidate index (returns None)
    best_cost : None
        Placeholder for best cost value (returns None)
    """
    # Ensure we have at least 2 candidates (Trap and Optimal)
    if m_candidates < 2:
        raise ValueError("m_candidates must be at least 2")
    
    # Calculate split sizes
    n_group1 = n_voters // 2
    n_group2 = n_voters - n_group1
    
    # Initialize voter coordinates array
    voter_coords = np.zeros((n_voters, 2), dtype=float)
    
    # Voter Group 1 (The Trap Sympathizers): Place at [0.49, 0.5]
    voter_coords[:n_group1, 0] = 0.49
    voter_coords[:n_group1, 1] = 0.5
    
    # Voter Group 2 (The Rational Block): Place at [1.0, 0.5]
    voter_coords[n_group1:, 0] = 1.0
    voter_coords[n_group1:, 1] = 0.5
    
    # Initialize candidate coordinates array
    candidate_coords = np.zeros((m_candidates, 2), dtype=float)
    
    # Candidate 0 (The Trap): Place at [0.0, 0.5]
    candidate_coords[0, 0] = 0.0
    candidate_coords[0, 1] = 0.5
    
    # Candidate 1 (The Optimal): Place at [1.0, 0.5]
    candidate_coords[1, 0] = 1.0
    candidate_coords[1, 1] = 0.5
    
    # Dummy Candidates: Place remaining m-2 candidates far away
    # Randomly between [2.0, 2.0] and [3.0, 3.0] so they receive 0 votes
    if m_candidates > 2:
        np.random.seed(42)  # Fixed seed for reproducibility
        for i in range(2, m_candidates):
            candidate_coords[i, 0] = np.random.uniform(2.0, 3.0)
            candidate_coords[i, 1] = np.random.uniform(2.0, 3.0)
    
    return voter_coords, candidate_coords, None, None

