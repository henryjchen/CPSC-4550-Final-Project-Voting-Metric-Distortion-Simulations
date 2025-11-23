"""
Test suite for Maximal Lotteries (ML) algorithm.

The Condorcet Winner Test is the gold standard for confirming the core logic
of the Maximal Lotteries implementation, as ML must assign 100% probability
to the Condorcet Winner if one exists.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import algorithms as algos


def test_condorcet_winner_3_candidates():
    """
    Test that ML assigns 100% probability to a Condorcet winner in a 3-candidate scenario.
    
    Setup: Candidate 0 is the Condorcet winner (beats both 1 and 2 in pairwise comparisons).
    Expected: ML distribution should assign probability 1.0 to candidate 0.
    """
    # Create rankings where candidate 0 beats both 1 and 2
    # With 3 voters, we ensure candidate 0 wins all pairwise comparisons
    rankings = np.array([
        [0, 1, 2],  # Voter 1: prefers 0 > 1 > 2
        [0, 2, 1],  # Voter 2: prefers 0 > 2 > 1
        [0, 1, 2],  # Voter 3: prefers 0 > 1 > 2
    ])
    
    # Verify Condorcet winner property: candidate 0 beats both 1 and 2
    s = algos.compute_pairwise_matrix(rankings)
    assert s[0, 1] > 0.5, "Candidate 0 should beat candidate 1"
    assert s[0, 2] > 0.5, "Candidate 0 should beat candidate 2"
    
    # Get ML distribution
    p_dist = algos.run_maximal_lotteries(rankings)
    
    # Verify: candidate 0 should have probability 1.0 (or very close due to floating point)
    assert np.isclose(p_dist[0], 1.0, atol=1e-6), \
        f"Condorcet winner (candidate 0) should have probability 1.0, got {p_dist[0]}"
    
    # Verify: other candidates should have probability 0 (or very close)
    assert np.isclose(p_dist[1], 0.0, atol=1e-6), \
        f"Non-winner (candidate 1) should have probability 0.0, got {p_dist[1]}"
    assert np.isclose(p_dist[2], 0.0, atol=1e-6), \
        f"Non-winner (candidate 2) should have probability 0.0, got {p_dist[2]}"
    
    # Verify: probabilities sum to 1
    assert np.isclose(np.sum(p_dist), 1.0, atol=1e-6), \
        f"Probabilities should sum to 1.0, got {np.sum(p_dist)}"
    
    print("CONFIRMED: Test passed: 3-candidate Condorcet winner")


def test_condorcet_winner_5_candidates():
    """
    Test that ML assigns 100% probability to a Condorcet winner in a 5-candidate scenario.
    
    Setup: Candidate 0 is the Condorcet winner (beats all others).
    Expected: ML distribution should assign probability 1.0 to candidate 0.
    """
    # Create rankings where candidate 0 is always first (Condorcet winner)
    # With 5 voters, we ensure candidate 0 wins all pairwise comparisons
    rankings = np.array([
        [0, 1, 2, 3, 4],  # Voter 1: 0 > 1 > 2 > 3 > 4
        [0, 2, 1, 4, 3],  # Voter 2: 0 > 2 > 1 > 4 > 3
        [0, 3, 4, 1, 2],  # Voter 3: 0 > 3 > 4 > 1 > 2
        [0, 4, 3, 2, 1],  # Voter 4: 0 > 4 > 3 > 2 > 1
        [0, 1, 3, 2, 4],  # Voter 5: 0 > 1 > 3 > 2 > 4
    ])
    
    # Verify Condorcet winner property: candidate 0 beats all others
    s = algos.compute_pairwise_matrix(rankings)
    for i in range(1, 5):
        assert s[0, i] > 0.5, f"Candidate 0 should beat candidate {i}"
    
    # Get ML distribution
    p_dist = algos.run_maximal_lotteries(rankings)
    
    # Verify: candidate 0 should have probability 1.0
    assert np.isclose(p_dist[0], 1.0, atol=1e-6), \
        f"Condorcet winner (candidate 0) should have probability 1.0, got {p_dist[0]}"
    
    # Verify: all other candidates should have probability 0
    for i in range(1, 5):
        assert np.isclose(p_dist[i], 0.0, atol=1e-6), \
            f"Non-winner (candidate {i}) should have probability 0.0, got {p_dist[i]}"
    
    # Verify: probabilities sum to 1
    assert np.isclose(np.sum(p_dist), 1.0, atol=1e-6), \
        f"Probabilities should sum to 1.0, got {np.sum(p_dist)}"
    
    print("CONFIRMED: Test passed: 5-candidate Condorcet winner")


def test_condorcet_winner_larger_election():
    """
    Test that ML assigns 100% probability to a Condorcet winner in a larger election.
    
    Setup: Candidate 0 is the Condorcet winner with more voters.
    Expected: ML distribution should assign probability 1.0 to candidate 0.
    """
    n_voters = 100
    m_candidates = 4
    
    # Create rankings where candidate 0 is always first
    rankings = np.zeros((n_voters, m_candidates), dtype=int)
    for v in range(n_voters):
        # Candidate 0 is always first
        rankings[v, 0] = 0
        # Other candidates in random order
        other_candidates = np.random.permutation(range(1, m_candidates))
        rankings[v, 1:] = other_candidates
    
    # Verify Condorcet winner property
    s = algos.compute_pairwise_matrix(rankings)
    for i in range(1, m_candidates):
        assert s[0, i] == 1.0, f"Candidate 0 should beat candidate {i} with 100% of votes"
    
    # Get ML distribution
    p_dist = algos.run_maximal_lotteries(rankings)
    
    # Verify: candidate 0 should have probability 1.0
    assert np.isclose(p_dist[0], 1.0, atol=1e-6), \
        f"Condorcet winner (candidate 0) should have probability 1.0, got {p_dist[0]}"
    
    # Verify: all other candidates should have probability 0
    for i in range(1, m_candidates):
        assert np.isclose(p_dist[i], 0.0, atol=1e-6), \
            f"Non-winner (candidate {i}) should have probability 0.0, got {p_dist[i]}"
    
    print("CONFIRMED: Test passed: Larger election Condorcet winner")


def test_no_condorcet_winner():
    """
    Test that ML still works correctly when no Condorcet winner exists.
    
    Setup: Create a Condorcet cycle (rock-paper-scissors scenario).
    Expected: ML should return a valid probability distribution (sums to 1, all non-negative).
    """
    # Create a Condorcet cycle: 0 beats 1, 1 beats 2, 2 beats 0
    # This is a classic "rock-paper-scissors" scenario
    rankings = np.array([
        [0, 1, 2],  # Voter 1: prefers 0 > 1 > 2
        [1, 2, 0],  # Voter 2: prefers 1 > 2 > 0
        [2, 0, 1],  # Voter 3: prefers 2 > 0 > 1
    ])
    
    # Verify no Condorcet winner exists
    s = algos.compute_pairwise_matrix(rankings)
    # Check that no candidate beats all others
    has_condorcet = False
    for c in range(3):
        if s[c, (c+1)%3] > 0.5 and s[c, (c+2)%3] > 0.5:
            has_condorcet = True
            break
    assert not has_condorcet, "This scenario should not have a Condorcet winner"
    
    # Get ML distribution
    p_dist = algos.run_maximal_lotteries(rankings)
    
    # Verify: probabilities sum to 1
    assert np.isclose(np.sum(p_dist), 1.0, atol=1e-6), \
        f"Probabilities should sum to 1.0, got {np.sum(p_dist)}"
    
    # Verify: all probabilities are non-negative
    assert np.all(p_dist >= -1e-6), \
        f"All probabilities should be non-negative, got {p_dist}"
    
    # Verify: no single candidate has probability 1.0 (since no Condorcet winner)
    assert not np.any(np.isclose(p_dist, 1.0, atol=1e-6)), \
        "No candidate should have probability 1.0 when no Condorcet winner exists"
    
    print("CONFIRMED: Test passed: No Condorcet winner (cycle scenario)")


def run_all_tests():
    """Run all Condorcet winner tests."""
    print("=" * 60)
    print("Running Maximal Lotteries Condorcet Winner Tests")
    print("=" * 60)
    print()
    
    try:
        test_condorcet_winner_3_candidates()
        test_condorcet_winner_5_candidates()
        test_condorcet_winner_larger_election()
        test_no_condorcet_winner()
        
        print()
        print("=" * 60)
        print("CONFIRMED: All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        raise
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Unexpected error: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()

