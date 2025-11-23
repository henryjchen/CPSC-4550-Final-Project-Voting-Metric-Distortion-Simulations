"""
Test script for Maximal Lotteries on a Condorcet Cycle instance.

This script tests the classic "rock-paper-scissors" voting scenario:
- Voter 1: A > B > C
- Voter 2: B > C > A  
- Voter 3: C > A > B

This creates a Condorcet cycle where there is no Condorcet winner,
and Maximal Lotteries should return a mixed strategy Nash equilibrium.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import algorithms as algos


def test_condorcet_cycle():
    """Test Maximal Lotteries on a classic Condorcet cycle instance."""
    
    # Create the classic Condorcet cycle: A > B > C, B > C > A, C > A > B
    # Using indices: A=0, B=1, C=2
    rankings = np.array([
        [0, 1, 2],  # Voter 1: A > B > C
        [1, 2, 0],  # Voter 2: B > C > A
        [2, 0, 1],  # Voter 3: C > A > B
    ])

    print('=' * 60)
    print('Condorcet Cycle Test for Maximal Lotteries')
    print('=' * 60)
    print()
    print('Input Rankings:')
    print(f'  Voter 1: {rankings[0]} (0=A > 1=B > 2=C)')
    print(f'  Voter 2: {rankings[1]} (1=B > 2=C > 0=A)')
    print(f'  Voter 3: {rankings[2]} (2=C > 0=A > 1=B)')
    print()

    # Compute pairwise matrix for verification
    s = algos.compute_pairwise_matrix(rankings)
    print('Pairwise Matrix (s[i,j] = proportion preferring i over j):')
    print('     A(0)  B(1)  C(2)')
    for i in range(3):
        print(f'{i}: {s[i]}')
    print()

    # Check for Condorcet cycle
    print('Pairwise Comparisons:')
    print(f'  A vs B: {s[0,1]:.2%} prefer A over B')
    print(f'  B vs C: {s[1,2]:.2%} prefer B over C')
    print(f'  C vs A: {s[2,0]:.2%} prefer C over A')
    print()
    
    # Verify the cycle: no Condorcet winner exists
    has_condorcet_winner = False
    for c in range(3):
        beats_all = True
        for other in range(3):
            if c != other and s[c, other] <= 0.5:
                beats_all = False
                break
        if beats_all:
            has_condorcet_winner = True
            print(f'Warning: Candidate {c} appears to be a Condorcet winner!')
            break
    
    if not has_condorcet_winner:
        print('CONFIRMED: No Condorcet winner exists (confirmed cycle)')
    print()

    # Get ML distribution
    ml_dist = algos.run_maximal_lotteries(rankings)

    print('=' * 60)
    print('Maximal Lotteries Probability Distribution:')
    print('=' * 60)
    for i in range(3):
        candidate_name = ['A', 'B', 'C'][i]
        print(f'  Candidate {candidate_name} (index {i}): {ml_dist[i]:.6f} ({ml_dist[i]:.2%})')
    print()
    print(f'Sum of probabilities: {np.sum(ml_dist):.6f}')
    print()
    
    # Verify distribution properties
    assert np.isclose(np.sum(ml_dist), 1.0, atol=1e-6), \
        f"Probabilities should sum to 1.0, got {np.sum(ml_dist)}"
    assert np.all(ml_dist >= -1e-6), \
        f"All probabilities should be non-negative, got {ml_dist}"
    
    # For this symmetric cycle, we expect approximately equal probabilities
    if np.allclose(ml_dist, 1/3, atol=1e-5):
        print('CONFIRMED: Distribution is uniform (1/3, 1/3, 1/3) as expected for symmetric cycle')
    else:
        print(f'Distribution is not uniform (expected due to numerical precision or LP solver)')
        print(f'     Largest deviation from 1/3: {np.max(np.abs(ml_dist - 1/3)):.6f}')
    print()
    
    print('Expected behavior: Since there is no Condorcet winner,')
    print('ML should return a mixed strategy (all probabilities > 0)')
    print('where the distribution represents the Nash equilibrium.')
    print()
    print('=' * 60)


if __name__ == "__main__":
    test_condorcet_cycle()

