# Metric Distortion Simulation Suite

**Course:** CPSC 4550: Algorithmic Game Theory  
**Author:** Henry Chen (henry.chen@yale.edu)  
**Date:** November 23, 2025

This repository contains the codebase used to generate the simulation results for my final paper (submitted on Gradescope)

## Repository Structure

The repository is divided into two distinct modules corresponding to the experimental sections of the paper.

### FOLDER: `Experiments 1-5` (Synthetic Simulations)
This folder contains the custom Python implementation of the voting algorithms (Maximal Lotteries, RaDiUS, Mixed Rule) and the synthetic testing suite. This was all generated solely and originally for this project. It generates the results for:
* **Experiment 1:** Average-Case Efficiency (Uniform Distribution).
* **Experiment 2:** Scalability (Candidate Count Sweep).
* **Experiment 3:** The RD-Trap (Outlier Vulnerability).
* **Experiment 4:** The ML-Trap (Majority Tyranny).
* **Experiment 5:** Noise Robustness (Plackett-Luce Sweep).

**How to Run:**
Navigate into the `Experiments 1-5` folder and follow the instructions in the local `README.md`.

---

### FOLDER: `Experiment 6` (Real-World Validation)
This folder contains the code for the "worst case distortion" analysis on the Spotify Daily dataset. It builds upon the replication codebase provided by Frank and Lederer (2025).

* **Base Code:** The root of this folder contains the original replication scripts for Frank and Lederer's average-case analysis (obtained from their paper's reference/appendix).
* **My Contribution:** The folder `worst_case_empirical_analysis_extension` contains the custom Linear Programming oracle and analysis scripts used to hunt for worst-case outliers in the dataset.

**Prerequisites:**
1.  You must download the **Spotify Daily Dataset (00047)** from [PrefLib](https://preflib.github.io/PrefLib-Jekyll/dataset/00047).
2.  Place the `.soc` files into the directory specified in the sub-folder README (i.e., spotifyday).

**How to Run:**
Navigate to `Experiment 6/worst_case_empirical_analysis_extension` and follow the instructions in the local `README.md` to reproduce the Black Swan results.

---

## Contact
If you have any questions regarding the implementation or reproduction of these results, please contact me at [henry.chen@yale.edu](mailto:henry.chen@yale.edu).
