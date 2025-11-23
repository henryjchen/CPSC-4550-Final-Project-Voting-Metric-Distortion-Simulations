"""
Phase 3: Worst Case Empirical Analysis
Generate specific numbers and plots that prove the thesis about worst-case distortion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports (if needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(csv_file=None):
    """Load the results CSV file"""
    if csv_file is None:
        # Default to results file in the outputs directory
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        csv_file = os.path.join(outputs_dir, "spotify_worst_case_empirical_results.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    return df


def peak_risk_analysis(df):
    """
    Calculate and display peak risk analysis table.
    
    Outputs:
    - Maximum distortion observed for each algorithm
    - Mean distortion for comparison
    """
    print("="*70)
    print("Peak Risk Analysis")
    print("="*70)
    
    algorithms = ['ML', 'RaDiUS', 'Mixed', 'RD']
    
    results = []
    for algo in algorithms:
        if algo in df.columns:
            values = df[algo].dropna()
            if len(values) > 0:
                results.append({
                    'Algorithm': algo,
                    'Max Distortion': f"{values.max():.4f}",
                    'Mean Distortion': f"{values.mean():.4f}",
                    'Min Distortion': f"{values.min():.4f}",
                    'Std Dev': f"{values.std():.4f}",
                    'Success Rate': f"{100 * len(values) / len(df):.1f}%"
                })
    
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*70)
    
    return summary_df


def insurance_check_plot(df):
    """
    Create scatter plot: ML Distortion vs Mixed Rule Distortion.
    
    Interpretation:
    - Points below y=x line: Days where Mixed Rule 'saved' us from ML's failure
    - Points above y=x line: Days where we paid the 'price of insurance'
    - Highlight the point with highest ML distortion (ML's worst day)
    """
    if 'ML' not in df.columns or 'Mixed' not in df.columns:
        print("ERROR: Missing ML or Mixed columns in data")
        return
    
    # Filter out NaN values
    valid_data = df[['ML', 'Mixed']].dropna()
    
    if len(valid_data) == 0:
        print("ERROR: No valid data points for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(valid_data['ML'], valid_data['Mixed'], 
              alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # Draw y=x line (red dashed)
    max_val = max(valid_data['ML'].max(), valid_data['Mixed'].max())
    min_val = min(valid_data['ML'].min(), valid_data['Mixed'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='y=x (Equal Performance)', alpha=0.7)
    
    # Highlight ML's worst day
    worst_ml_idx = valid_data['ML'].idxmax()
    worst_ml_ml = valid_data.loc[worst_ml_idx, 'ML']
    worst_ml_mixed = valid_data.loc[worst_ml_idx, 'Mixed']
    
    ax.scatter([worst_ml_ml], [worst_ml_mixed], 
              s=300, c='red', marker='*', 
              edgecolors='black', linewidth=2, 
              label=f"ML's Worst Day (ML={worst_ml_ml:.4f}, Mixed={worst_ml_mixed:.4f})",
              zorder=5)
    
    # Labels and title
    ax.set_xlabel('Maximal Lotteries Distortion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mixed Rule (CRWW) Distortion', fontsize=12, fontweight='bold')
    ax.set_title('The Insurance Check: ML vs Mixed Rule\n' + 
                'Points below line = "Insurance paid off", Points above = "Price of insurance"',
                fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot to outputs directory
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, "worst_case_empirical_insurance_check.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nInsurance Check plot saved to: {output_file}")
    
    # Calculate statistics
    below_line = (valid_data['Mixed'] < valid_data['ML']).sum()
    above_line = (valid_data['Mixed'] > valid_data['ML']).sum()
    on_line = (valid_data['Mixed'] == valid_data['ML']).sum()
    
    print(f"\nInsurance Check Statistics:")
    print(f"  Days where Mixed 'saved' us (below line): {below_line} ({100*below_line/len(valid_data):.1f}%)")
    print(f"  Days where we paid 'price' (above line): {above_line} ({100*above_line/len(valid_data):.1f}%)")
    print(f"  Days with equal performance: {on_line} ({100*on_line/len(valid_data):.1f}%)")
    
    plt.show()
    
    return fig


def timeline_plot(df):
    """
    Plot Distortion vs Date for ML and Mixed Rule to see if spikes correlate.
    """
    if 'ML' not in df.columns or 'Mixed' not in df.columns or 'DateIndex' not in df.columns:
        print("ERROR: Missing required columns for timeline plot")
        return
    
    # Filter out NaN values
    valid_data = df[['DateIndex', 'ML', 'Mixed']].dropna()
    valid_data = valid_data.sort_values('DateIndex')
    
    if len(valid_data) == 0:
        print("ERROR: No valid data points for timeline plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both algorithms
    ax.plot(valid_data['DateIndex'], valid_data['ML'], 
           'b-', linewidth=1.5, alpha=0.7, label='Maximal Lotteries (ML)')
    ax.plot(valid_data['DateIndex'], valid_data['Mixed'], 
           'g-', linewidth=1.5, alpha=0.7, label='Mixed Rule (CRWW)')
    
    # Highlight ML's worst day
    worst_ml_idx = valid_data['ML'].idxmax()
    worst_date = valid_data.loc[worst_ml_idx, 'DateIndex']
    worst_ml_val = valid_data.loc[worst_ml_idx, 'ML']
    worst_mixed_val = valid_data.loc[worst_ml_idx, 'Mixed']
    
    ax.scatter([worst_date], [worst_ml_val], 
              s=200, c='red', marker='*', 
              edgecolors='black', linewidth=2, 
              label=f"ML's Worst Day (ML={worst_ml_val:.4f})",
              zorder=5)
    ax.scatter([worst_date], [worst_mixed_val], 
              s=200, c='orange', marker='*', 
              edgecolors='black', linewidth=2, 
              label=f"Mixed on ML's Worst Day ({worst_mixed_val:.4f})",
              zorder=5)
    
    # Labels and title
    ax.set_xlabel('Day Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Worst-Case Distortion', fontsize=12, fontweight='bold')
    ax.set_title('Timeline: Worst-Case Distortion Over Time\n' +
                'ML vs Mixed Rule - Do spikes correlate?',
                fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot to outputs directory
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, "worst_case_empirical_timeline.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timeline plot saved to: {output_file}")
    
    plt.show()
    
    return fig


def detailed_worst_day_analysis(df):
    """Analyze ML's worst day in detail"""
    if 'ML' not in df.columns:
        print("ERROR: ML column not found")
        return
    
    worst_ml_idx = df['ML'].idxmax()
    worst_day = df.loc[worst_ml_idx]
    
    print("\n" + "="*70)
    print("ML's Worst Day Analysis")
    print("="*70)
    print(f"Date: {worst_day.get('Date', 'N/A')}")
    print(f"Date Index: {worst_day.get('DateIndex', 'N/A')}")
    print(f"\nDistortions on this day:")
    for algo in ['ML', 'RaDiUS', 'Mixed', 'RD']:
        if algo in worst_day:
            val = worst_day[algo]
            if pd.notna(val):
                print(f"  {algo:8s}: {val:.4f}")
    
    # Calculate "insurance value"
    if 'Mixed' in worst_day and pd.notna(worst_day['Mixed']):
        ml_val = worst_day['ML']
        mixed_val = worst_day['Mixed']
        improvement = ml_val - mixed_val
        improvement_pct = 100 * improvement / ml_val if ml_val > 0 else 0
        print(f"\nInsurance Value:")
        print(f"  ML Distortion: {ml_val:.4f}")
        print(f"  Mixed Distortion: {mixed_val:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}% reduction)")
    
    print("="*70)


def main():
    """Main analysis function"""
    print("="*70)
    print("Worst Case Empirical Analysis: Worst-Case Distortion Study")
    print("="*70)
    
    # Load data
    print("\nLoading results...")
    df = load_results()  # Uses default path
    print(f"Loaded {len(df)} days of data")
    
    # Peak Risk Analysis
    summary_df = peak_risk_analysis(df)
    
    # Insurance Check Plot
    print("\n" + "="*70)
    print("Generating Insurance Check Plot...")
    print("="*70)
    insurance_check_plot(df)
    
    # Timeline Plot
    print("\n" + "="*70)
    print("Generating Timeline Plot...")
    print("="*70)
    timeline_plot(df)
    
    # Detailed Worst Day Analysis
    detailed_worst_day_analysis(df)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

