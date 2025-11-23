import matplotlib.pyplot as plt
import numpy as np
import os

def plot_noise_analysis(results, temperatures):
    """
    Generates and displays the line plot for Experiment 5: Distortion vs. Noise Temperature.
    
    Args:
        results (dict): Dictionary mapping algorithm names to lists of average distortions.
                       Keys should be algorithm names (e.g., 'Random Dictatorship', 'RaDiUS').
                       Values should be lists of distortion values corresponding to temperatures.
        temperatures (list): List of noise temperature values (tau) used in the experiment.
        
    Returns:
        None: Displays the plot and saves it to output/noise_analysis.png.
    """
    plt.figure(figsize=(10, 6))
    
    ALGO_STYLES = {
        'Random Dictatorship': ('#4CAF50', 's'),
        'RaDiUS': ('#FF9800', 'o'),
        'Maximal Lotteries': ('#F44336', '^'),
        'Mixed Rule': ('#2196F3', 'D')
    }

    for algo_name, distortions in results.items():
        if algo_name in ALGO_STYLES:
            color, marker = ALGO_STYLES[algo_name]
            plt.plot(
                temperatures,
                distortions,
                label=algo_name,
                color=color,
                marker=marker,
                linestyle='-',
                linewidth=2
            )

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Perfect (1.0)')
    
    plt.title('Experiment 5: Algorithm Robustness to Voter Noise ($\u03C4$)', fontsize=16)
    plt.xlabel('Noise Temperature ($\u03C4$): Voter Irrationality', fontsize=12)
    plt.ylabel('Average Distortion (Lower is Better)', fontsize=12)
    
    plt.xscale('log') 
    
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which="both", linestyle=':', linewidth=0.5)
    
    max_distortion = max(max(d) for d in results.values())
    plt.ylim(0.95, max_distortion * 1.05)
    
    plt.tight_layout()
    
    # Save the figure to the output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "noise_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved noise analysis plot to {output_path}")
    
    plt.show()


def plot_candidate_count_analysis(results, candidate_counts):
    """
    Generates and displays the line plot for Experiment 2: Distortion vs. Candidate Count (M).
    
    Args:
        results (dict): Dictionary mapping algorithm names to lists of average distortions.
                       Keys should be algorithm names (e.g., 'Random Dictatorship', 'RaDiUS').
                       Values should be lists of distortion values corresponding to candidate counts.
        candidate_counts (list): List of candidate count values (M) used in the experiment.
        
    Returns:
        None: Displays the plot and saves it to output/candidate_count_analysis.png.
    """
    plt.figure(figsize=(10, 6))
    
    ALGO_STYLES = {
        'Random Dictatorship': ('#4CAF50', 's'),
        'RaDiUS': ('#FF9800', 'o'),
        'Maximal Lotteries': ('#F44336', '^'),
        'Mixed Rule': ('#2196F3', 'D')
    }

    for algo_name, distortions in results.items():
        if algo_name in ALGO_STYLES:
            color, marker = ALGO_STYLES[algo_name]
            plt.plot(
                candidate_counts,
                distortions,
                label=algo_name,
                color=color,
                marker=marker,
                linestyle='-',
                linewidth=2
            )

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Perfect (1.0)')
    
    plt.title('Experiment 2: Algorithm Scalability (Distortion vs. Candidate Count M)', fontsize=16)
    plt.xlabel('Number of Candidates (M)', fontsize=12)
    plt.ylabel('Average Distortion (Lower is Better)', fontsize=12)
    
    plt.xticks(candidate_counts)
    
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which="both", linestyle=':', linewidth=0.5)
    
    max_distortion = max(max(d) for d in results.values())
    plt.ylim(0.95, max_distortion * 1.05)
    
    plt.tight_layout()
    
    # Save the figure to the output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "candidate_count_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved candidate count analysis plot to {output_path}")
    
    plt.show()