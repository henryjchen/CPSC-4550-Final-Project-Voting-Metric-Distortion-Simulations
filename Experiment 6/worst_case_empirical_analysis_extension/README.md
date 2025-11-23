# Worst Case Empirical Analysis: Worst-Case Distortion Study

**Location**: This extension is located in the `worst_case_empirical_analysis_extension/` directory.

This implementation provides a three-phase system for analyzing worst-case distortion on real-world Spotify preference data.

## Overview

**Goal**: Prove that worst-case distortion analysis reveals worst-case events where algorithms fail catastrophically, and demonstrate that mixed rules provide "insurance" against such failures.

**Approach**: 
- Phase 1: Distortion Oracle (worst-case assumption)
- Phase 2: Parallelized execution engine
- Phase 3: Worst case empirical analysis and visualization

## Prerequisites

1. **Spotify Dataset**: Download from PrefLib #00047
   - URL: https://preflib.github.io/PrefLib-Jekyll/dataset/00047
   - Extract to `./spotifyday/` directory

2. **Dependencies**: 
   ```bash
   pip install preflibtools numpy pandas matplotlib pulp
   ```

3. **Solver**: The code uses Gurobi by default. If you don't have a license:
   - Change `pl.GUROBI(msg=False).solve(model)` to `pl.PULP_CBC_CMD(msg=False).solve(model)` in `Helper.py` line 70
   - CBC is free and comes with PuLP, but is slower

## Phase 1: Distortion Oracle

**File**: `Helper.py` (function `get_worst_case_distortion`)

**What it does**:
- Takes a preference profile and a probability distribution (lottery)
- For each candidate, assumes it's the true optimal
- Computes distortion for each assumption
- Returns the maximum (worst-case) distortion

**Usage**:
```python
from Helper import get_worst_case_distortion
distortion = get_worst_case_distortion(profile, lottery)
```

**Philosophy**: "Nature is adversarial" - we assume the ground truth metric space is whatever makes our algorithm look worst.

## Phase 2: Execution Engine

**File**: `run_spotify_experiment.py`

**What it does**:
- Loads all Spotify profiles from `./spotifyday/`
- Runs 4 algorithms on each profile:
  - **ML**: Maximal Lotteries (C2ML)
  - **RaDiUS**: Radius-based algorithm
  - **Mixed**: CRWW (convex combination of ML and RaDiUS)
  - **RD**: Random Dictatorship
- Computes worst-case distortion for each algorithm
- Uses multiprocessing for parallel execution
- Saves results to `spotify_worst_case_empirical_results.csv`

**Usage**:
```bash
cd worst_case_empirical_analysis_extension
python run_spotify_experiment.py
```

Or from the project root:
```bash
python worst_case_empirical_analysis_extension/run_spotify_experiment.py
```

**Output**: `spotify_worst_case_empirical_results.csv` with columns:
- `Date`: Date identifier
- `DateIndex`: Sequential index
- `FilePath`: Path to source file
- `ML`: Worst-case distortion for Maximal Lotteries
- `RaDiUS`: Worst-case distortion for RaDiUS
- `Mixed`: Worst-case distortion for Mixed Rule
- `RD`: Worst-case distortion for Random Dictatorship

**Performance**: 
- Processes ~365 days × 4 algorithms × ~20 candidates = ~29,000 LPs
- Uses all available CPU cores
- Progress is printed to console

## Phase 3: Worst Case Empirical Analysis

**File**: `analyze_worst_case_empirical.py`

**What it does**:
1. **Peak Risk Analysis Table**: 
   - Maximum distortion for each algorithm
   - Mean distortion for comparison
   - Standard deviation and success rates

2. **Insurance Check Plot**:
   - Scatter plot: ML Distortion (x-axis) vs Mixed Rule Distortion (y-axis)
   - Red dashed line at y=x
   - Points below line = "Insurance paid off" (Mixed saved us)
   - Points above line = "Price of insurance" (Mixed performed worse)
   - Highlights ML's worst day with a red star

3. **Timeline Plot**:
   - Distortion over time for ML and Mixed Rule
   - Shows if spikes correlate
   - Highlights worst day

4. **Detailed Worst Day Analysis**:
   - In-depth look at ML's worst day
   - Shows all algorithm performances
   - Calculates "insurance value" (improvement from Mixed)

**Usage**:
```bash
cd worst_case_empirical_analysis_extension
python analyze_worst_case_empirical.py
```

Or from the project root:
```bash
python worst_case_empirical_analysis_extension/analyze_worst_case_empirical.py
```

**Outputs**:
- Console: Tables and statistics
- `worst_case_empirical_insurance_check.png`: Scatter plot
- `worst_case_empirical_timeline.png`: Timeline plot

## Workflow

1. **Prepare Data**:
   ```bash
   # Download Spotify dataset to ./spotifyday/
   # Verify files exist
   ls spotifyday/*.soc
   ```

2. **Run Experiment**:
   ```bash
   python run_spotify_experiment.py
   ```
   This may take a while (hours for full dataset with CBC solver).

3. **Analyze Results**:
   ```bash
   python analyze_worst_case_empirical.py
   ```

## Expected Results

The analysis should reveal:

1. **Peak Risk**: ML can have very high worst-case distortion on certain days
2. **Insurance Value**: Mixed Rule typically has lower worst-case distortion
3. **Worst Case Events**: Specific days where ML fails catastrophically
4. **Trade-off**: Some days where Mixed performs slightly worse (price of insurance)

## Troubleshooting

**Issue**: "GUROBI: Not Available"
- **Solution**: Change solver in `Helper.py` line 70 to `pl.PULP_CBC_CMD(msg=False).solve(model)`

**Issue**: "No profiles found"
- **Solution**: Verify `./spotifyday/` directory exists and contains `.soc` files

**Issue**: "Model too large for size-limited license"
- **Solution**: Use CBC solver (free) or get full Gurobi license

**Issue**: Memory errors
- **Solution**: Reduce number of workers in `run_spotify_experiment.py` (change `n_workers`)

## Files Created

- `spotify_worst_case_empirical_results.csv`: Raw results data
- `worst_case_empirical_insurance_check.png`: Scatter plot
- `worst_case_empirical_timeline.png`: Timeline plot

## Notes

- The worst-case distortion represents the "adversarial nature" assumption
- This is more pessimistic than average-case analysis
- The Mixed Rule (CRWW) combines ML and RaDiUS to provide robustness
- Results may vary depending on solver used (Gurobi vs CBC)

