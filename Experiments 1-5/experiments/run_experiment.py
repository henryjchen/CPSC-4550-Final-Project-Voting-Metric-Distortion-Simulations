"""
This script serves as the main entry point for running a suite of experiments
on metric voting algorithms. It is designed to measure and compare the distortion
of various algorithms under different conditions, such as random voter/candidate
placements, specific "trap" scenarios, and varying levels of noise.

The script supports running individual experiments or all of them in sequence.
Results are printed to the console in summary tables, and for some experiments,
graphs are generated and saved to disk.

Usage:
    python run_experiment.py [experiment_name]

Available experiments:
    - uniform: Baseline performance in a uniform random environment.
    - rd_trap: Tests robustness against an outlier trap for RaDiUS.
    - ml_trap: Tests robustness against a Condorcet cycle trap for ML.
    - noise: Analyzes algorithm performance across different noise levels.
    - candidate_sweep: Measures scalability by varying the number of candidates.
    - all: Runs all of the above experiments.
"""
import sys
import numpy as np
from tqdm import tqdm
import argparse

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import generators
from utils import simulation_utils as sim_utils
from utils import noise_models
from algorithms import (
    run_radius,
    run_maximal_lotteries,
    run_random_dictatorship,
    run_mixed_rule
)
import algorithms as algos
from visualization import plotting

# --- Configuration ---
N_SIMULATIONS = 1000
N_VOTERS = 100
M_CANDIDATES = 5
BETA_FOR_RADIUS = 0.75  # Default beta for RaDiUS, from theoretical analysis in papers.

# Set this to True to print distortion for every single run
DEBUG_INDIVIDUAL_RUNS = False

# Note: Maximal Lotteries and Mixed Rule are handled separately in the simulation
# loop because they are probabilistic and require sampling.
ALGORITHMS = {
    "Random Dictatorship": algos.run_random_dictatorship,
    "RaDiUS": lambda r: algos.run_radius(r, beta=BETA_FOR_RADIUS),
}


# --- Core Simulation Logic ---

def run_single_simulation(generator_func, temperature, m_candidates_override=M_CANDIDATES, p_ml_override=None):
    """
    Executes a single end-to-end simulation run, from generating an instance
    to calculating the distortion of each algorithm.
    """
    m_candidates = m_candidates_override
    coords_v, coords_c, dist_matrix_pc, rankings_pc = generator_func(N_VOTERS, m_candidates)

    if dist_matrix_pc is not None:
        dist_matrix = dist_matrix_pc
        costs = sim_utils.get_social_costs(dist_matrix)
        ordinal_ballots = rankings_pc if temperature < 0.001 else noise_models.generate_noisy_rankings(dist_matrix, temperature)
    else:
        dist_matrix = sim_utils.get_distance_matrix(coords_v, coords_c)
        costs = sim_utils.get_social_costs(dist_matrix)
        ordinal_ballots = noise_models.generate_noisy_rankings(dist_matrix, temperature)

    optimal_winner = sim_utils.get_true_optimal_candidate(costs)
    optimal_cost = costs[optimal_winner]

    if optimal_cost == 0:
        return {name: 1.0 for name in list(ALGORITHMS.keys()) + ["Maximal Lotteries", "Mixed Rule"]}

    results = {}
    for name, func in ALGORITHMS.items():
        # All algorithms now return probability distributions
        dist = func(ordinal_ballots)
        # Sample a single winner from the distribution for fair comparison
        winner = np.random.choice(len(dist), p=dist)
        distortion = costs[winner] / optimal_cost if winner is not None else float('inf')
        results[name] = distortion

    # All algorithms now return distributions consistently
    ml_dist = algos.run_maximal_lotteries(ordinal_ballots)
    ml_winner = np.random.choice(len(ml_dist), p=ml_dist)
    results["Maximal Lotteries"] = costs[ml_winner] / optimal_cost

    mixed_dist = algos.run_mixed_rule(ordinal_ballots, p_ml_override=p_ml_override)
    mixed_winner = np.random.choice(len(mixed_dist), p=mixed_dist)
    results["Mixed Rule"] = costs[mixed_winner] / optimal_cost

    if DEBUG_INDIVIDUAL_RUNS:
        dist_str = ", ".join([f"C{i}={ml_dist[i]:.3f}" for i in range(min(5, len(ml_dist)))])
        print(f"Run {np.random.randint(10000)} - ML dist: [{dist_str}], ML distortion: {results['Maximal Lotteries']:.2f}, RaDiUS: {results['RaDiUS']:.2f}, Mixed: {results['Mixed Rule']:.2f}", file=sys.stderr)

    return results


def run_experiment_batch(experiment_name, generator_func, temperature, m_candidates_override=M_CANDIDATES, p_ml_override=None):
    """Runs a batch of simulations and returns the average distortion for each algorithm."""
    all_algo_names = list(ALGORITHMS.keys()) + ["Maximal Lotteries", "Mixed Rule"]
    all_distortions = {name: [] for name in all_algo_names}

    for _ in tqdm(range(N_SIMULATIONS), desc=f"Simulating {experiment_name}", disable=False, file=sys.stderr):
        run_results = run_single_simulation(generator_func, temperature, m_candidates_override, p_ml_override)
        for name, distortion in run_results.items():
            if name in all_distortions:
                all_distortions[name].append(distortion)

    avg_distortions = {name: np.mean(dists) for name, dists in all_distortions.items() if dists}
    return avg_distortions


def print_summary_table(header, avg_distortions):
    """Prints a clean summary of the average distortion results."""
    max_len = max(len(name) for name in avg_distortions.keys())
    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)
    for name, avg_dist in avg_distortions.items():
        print(f"{name.ljust(max_len)}: {avg_dist:.4f}")
    print("-" * 50)


# --- Experiment Definitions ---

def run_uniform_experiment():
    """Experiment 1: Checks baseline performance in a uniform random environment."""
    avg_distortions = run_experiment_batch(
        "UNIFORM Case (Average Performance)",
        generators.generate_instance,
        temperature=0.1
    )
    print_summary_table("Average Distortion Results (UNIFORM)", avg_distortions)


def run_rd_trap_experiment():
    """Experiment 3: Checks robustness against the N-1 vs 1 outlier trap."""
    avg_distortions = run_experiment_batch(
        "RD_TRAP Case (Outlier Trap)",
        generators.generate_rd_trap_instance,
        temperature=0.001
    )
    print_summary_table("Average Distortion Results (RD_TRAP)", avg_distortions)


def run_ml_trap_experiment():
    """Experiment 4: Checks robustness against the Condorcet cycle trap."""
    avg_distortions = run_experiment_batch(
        "ML_TRAP Case (Cycle Trap)",
        generators.generate_ml_trap_instance,
        temperature=0.001
    )
    print_summary_table("Average Distortion Results (ML_TRAP)", avg_distortions)


def run_noise_analysis():
    """Experiment 5: Generates data for the Distortion vs. Noise graph."""
    print("\n" + "=" * 50)
    print("Experiment 5: Distortion vs. Noise Analysis")
    print("=" * 50)

    NOISE_TEMPS = [0.01, 0.10, 0.50, 1.00, 5.00]
    algo_names = list(ALGORITHMS.keys()) + ["Maximal Lotteries", "Mixed Rule"]
    noise_results = {name: [] for name in algo_names}

    for temp in NOISE_TEMPS:
        avg_distortions = run_experiment_batch(
            f"Noise Temp: {temp:.2f}",
            generators.generate_instance,
            temperature=temp
        )
        for name, dist in avg_distortions.items():
            if name in noise_results:
                noise_results[name].append(dist)

    # Print results table
    header = "Algorithm".ljust(20) + " | " + " | ".join([f"{t:.2f}".ljust(12) for t in NOISE_TEMPS])
    print("\n--- Noise Analysis Table (Data for Graph) ---")
    print(header)
    print("-" * (len(header) + len(NOISE_TEMPS)))
    for name in algo_names:
        row_str = f"{name.ljust(20)} | " + "".join([f"{dist:.4f}".ljust(12) for dist in noise_results[name]])
        print(row_str)

    plotting.plot_noise_analysis(noise_results, NOISE_TEMPS)


def run_candidate_count_sweep():
    """Experiment 2: Measures scalability by sweeping the number of candidates (M)."""
    print("\n" + "=" * 50)
    print("Experiment 2: Scalability Analysis (Distortion vs. Candidate Count M)")
    print("=" * 50)

    M_COUNTS = [3, 5, 8, 12, 15]
    algo_names = list(ALGORITHMS.keys()) + ["Maximal Lotteries", "Mixed Rule"]
    sweep_results = {name: [] for name in algo_names}

    for M in M_COUNTS:
        avg_distortions = run_experiment_batch(
            f"Candidate Count M={M}",
            generators.generate_instance,
            temperature=0.01,
            m_candidates_override=M
        )
        for name, dist in avg_distortions.items():
            if name in sweep_results:
                sweep_results[name].append(dist)
        print(f"-> M={M}: ML={avg_distortions['Maximal Lotteries']:.3f}, RD={avg_distortions['Random Dictatorship']:.3f}")

    plotting.plot_candidate_count_analysis(sweep_results, M_COUNTS)



def main():
    """Main entry point to run selected experiments."""
    parser = argparse.ArgumentParser(description="Run metric voting distortion simulations.")
    parser.add_argument(
        "experiment",
        choices=["uniform", "rd_trap", "ml_trap", "noise", "candidate_sweep", "all"],
        default="all",
        nargs="?",
        help="The specific experiment to run (default: 'all')."
    )
    args = parser.parse_args()

    experiment_map = {
        "uniform": run_uniform_experiment,
        "rd_trap": run_rd_trap_experiment,
        "ml_trap": run_ml_trap_experiment,
        "noise": run_noise_analysis,
        "candidate_sweep": run_candidate_count_sweep,
    }

    if args.experiment == "all":
        for func in experiment_map.values():
            func()
    elif args.experiment in experiment_map:
        experiment_map[args.experiment]()


if __name__ == '__main__':
    sys.setrecursionlimit(2000)
    main()
