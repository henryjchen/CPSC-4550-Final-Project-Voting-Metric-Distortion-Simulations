# Metric Voting Algorithms: Distortion Analysis

This repository contains implementations and experimental analysis of metric voting algorithms, focusing on measuring and comparing the distortion (inefficiency) of various voting rules under different conditions.

## Overview

This project implements and evaluates several voting algorithms designed for metric spaces, where voters and candidates are positioned in a geometric space and preferences are derived from distances. The algorithms are tested under various scenarios including random placements, adversarial "trap" instances, and varying levels of voter noise.

## Implemented Algorithms

The repository implements four voting algorithms:

1. **Random Dictatorship**: A simple baseline that selects a random voter and returns their top choice.
2. **RaDiUS** (beta-Random Dictatorship on the Uncovered Set): Uses a beta threshold to compute the uncovered set, then samples a random voter's favorite from that set.
3. **Maximal Lotteries (ML)**: Finds a probability distribution over candidates that is a Nash equilibrium of a zero-sum game defined by pairwise comparisons.
4. **Mixed Rule**: Combines ML and RaDiUS by randomly selecting one algorithm according to a theoretically optimal probability distribution.

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see below)

### Dependencies

Install the required packages with the following versions:

```bash
pip matplotlib==3.9.2 networkx==3.2.1 numpy==2.0.2 scipy
```

## Project Structure

```
code/
├── algorithms/          # Voting algorithm implementations
│   ├── common.py       # Shared utilities (pairwise matrix computation)
│   ├── ml.py           # Maximal Lotteries algorithm
│   ├── radius.py       # RaDiUS algorithm
│   ├── random_dictatorship.py  # Random Dictatorship algorithm
│   └── mixed_rule.py   # Mixed Rule algorithm
├── data/               # Data generation and loading
│   └── generators.py   # Instance generators (uniform, traps, etc.)
├── utils/              # Utility functions
│   ├── simulation_utils.py  # Distance calculations, social costs
│   └── noise_models.py     # Plackett-Luce noise model
├── experiments/        # Experiment runner
│   └── run_experiment.py   # Main experiment script
├── visualization/      # Plotting functions
│   └── plotting.py     # Visualization utilities
├── tests/              # Test files
│   ├── test_ml_condorcet.py
│   └── test_condorcet_cycle.py
├── output/             # Generated plots and results (exp. 2 and 5)
```

## Usage

### Running Experiments

The main entry point is `experiments/run_experiment.py`. You can run individual experiments or all experiments:

```bash
# Run a specific experiment
python experiments/run_experiment.py uniform
python experiments/run_experiment.py rd_trap
python experiments/run_experiment.py ml_trap
python experiments/run_experiment.py noise
python experiments/run_experiment.py candidate_sweep

# Run all experiments
python experiments/run_experiment.py all
```

### Available Experiments

1. **Experiment 1: Uniform Baseline** (`uniform`)
   - Tests baseline performance in a uniform random environment
   - Generates random voter and candidate positions in [0,1]²
   - Temperature: 0.1

2. **Experiment 2: Candidate Count Sweep** (`candidate_sweep`)
   - Measures scalability by varying the number of candidates (M)
   - Tests with M ∈ {3, 5, 8, 12, 15}
   - Generates a plot: `output/candidate_count_analysis.png`

3. **Experiment 3: RaDiUS Trap** (`rd_trap`)
   - Tests robustness against the N-1 vs 1 outlier trap
   - Designed to challenge RaDiUS with an adversarial instance
   - Temperature: 0.001

4. **Experiment 4: ML Trap** (`ml_trap`)
   - Tests robustness against a "Majority Tyranny" trap
   - Creates a 51% vs 49% split where ML picks the inefficient Condorcet winner
   - Demonstrates ML's failure in polarized elections
   - Temperature: 0.001

5. **Experiment 5: Noise Analysis** (`noise`)
   - Analyzes algorithm performance across different noise levels
   - Tests temperatures: [0.01, 0.10, 0.50, 1.00, 5.00]
   - Generates a plot: `output/noise_analysis.png`

## Algorithm Details

### Maximal Lotteries (ML)

ML solves a zero-sum game where the payoff matrix is derived from pairwise comparisons. It finds a Nash equilibrium distribution over candidates using linear programming.

**Key Features:**
- Always selects a Condorcet winner with probability 1.0 if one exists
- Returns a probability distribution (mixed strategy)
- Uses scipy's linear programming solver

### RaDiUS

RaDiUS first computes the uncovered set based on a beta threshold, then samples a random voter and returns their favorite candidate from the uncovered set.

**Key Features:**
- Default beta: 0.75 (for standalone RaDiUS)
- Returns a degenerate distribution (probability 1.0 on winner)
- More robust to outliers than pure Random Dictatorship

### Mixed Rule

The Mixed Rule combines ML and RaDiUS by:
- Running ML with probability p_ml (computed from theoretical bounds)
- Running RaDiUS with probability (1 - p_ml), using a sampled beta value

**Key Features:**
- Uses B = 0.876353 as the upper bound for beta sampling
- Achieves better distortion bounds than either algorithm alone
- Beta values are sampled from [0.5, B] according to a theoretical distribution

## Output

### Console Output

Each experiment prints a summary table showing average distortion for each algorithm. Example below:

```
==================================================
Average Distortion Results (UNIFORM)
==================================================
Random Dictatorship: 1.1673
RaDiUS             : 1.1615
Maximal Lotteries  : 1.0235
Mixed Rule         : 1.0701
--------------------------------------------------
```

### Generated Plots

- `output/noise_analysis.png`: Experiment 5 - Distortion vs. Noise Temperature
- `output/candidate_count_analysis.png`: Experiment 2 - Distortion vs. Candidate Count

## Testing

Run the test suites to verify ML algorithm correctness:

```bash
# Test ML on Condorcet winners
python tests/test_ml_condorcet.py

# Test ML on Condorcet cycles
python tests/test_condorcet_cycle.py
```

## Configuration

Key configuration parameters in `experiments/run_experiment.py`:

- `N_SIMULATIONS = 1000`: Number of simulation runs per experiment
- `N_VOTERS = 100`: Default number of voters
- `M_CANDIDATES = 5`: Default number of candidates
- `BETA_FOR_RADIUS = 0.75`: Default beta for standalone RaDiUS

## Notes

- All algorithms return probability distributions over candidates
- Distortion is calculated as the ratio of the algorithm's social cost to the optimal social cost
- Lower distortion values indicate better performance (1.0 is perfect)
- The noise model uses Plackett-Luce with temperature parameter τ
