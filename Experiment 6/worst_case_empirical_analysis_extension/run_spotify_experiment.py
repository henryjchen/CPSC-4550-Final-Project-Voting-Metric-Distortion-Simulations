"""
Phase 2: The Execution Engine (Parallelized)
Run worst-case distortion analysis for every algorithm on every Spotify profile.
Uses multiprocessing to handle ~29,000 LPs efficiently.
"""

import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Sampling import CulturesRealWorld
from RSCFs import C2ML, CRWW, RD
from Helper import get_worst_case_distortion

# Regex pattern for numerical sorting
numbers = re.compile(r'(\d+)')


def load_spotify_profiles(data_dir="./spotifyday"):
    """
    Load all Spotify profiles from .soc files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing .soc files
        
    Returns:
    --------
    list : List of tuples (date_index, profile, filepath)
    """
    profiles = []
    
    # Find all .soc files
    soc_files = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if not os.path.isdir(file_path) and file.endswith(".soc"):
            soc_files.append(file_path)
    
    # Sort numerically
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    soc_files = sorted(soc_files, key=numericalSort)
    
    # Load each profile with progress bar
    for idx, filepath in enumerate(tqdm(soc_files, desc="Loading profiles", unit="file")):
        try:
            profile = CulturesRealWorld.getReal(filepath)
            # Extract date from filename (e.g., "00047-00000001" -> "00000001")
            date_str = os.path.basename(filepath).split('-')[-1].replace('.soc', '')
            profiles.append((idx, profile, date_str, filepath))
        except Exception as e:
            print(f"\nError loading {filepath}: {e}")
            continue
    
    return profiles


def run_ml(profile):
    """Run Maximal Lotteries (C2ML) algorithm"""
    try:
        return C2ML.computeC2ML(profile)
    except Exception as e:
        print(f"Error in ML: {e}")
        return None


def run_radius(profile):
    """Run RaDiUS algorithm"""
    try:
        n, m = np.shape(profile)
        radius_lottery = CRWW.compute_RaDiUS_integral(profile, m, n)
        # Normalize
        radius_lottery = radius_lottery / np.sum(radius_lottery)
        return radius_lottery.tolist()
    except Exception as e:
        print(f"Error in RaDiUS: {e}")
        return None


def run_mixed(profile):
    """Run CRWW (Mixed) algorithm"""
    try:
        return CRWW.compute_CCRW(profile)
    except Exception as e:
        print(f"Error in Mixed: {e}")
        return None


def run_rd(profile):
    """Run Random Dictatorship algorithm"""
    try:
        return RD.computeRD(profile)
    except Exception as e:
        print(f"Error in RD: {e}")
        return None


def process_day(data_tuple):
    """
    Worker function to process a single day's profile.
    
    Parameters:
    -----------
    data_tuple : tuple
        (date_index, profile, date_str, filepath)
        
    Returns:
    --------
    dict : Results dictionary with distortions for each algorithm
    """
    idx, profile, date_str, filepath = data_tuple
    
    result = {
        'Date': date_str,
        'DateIndex': idx,
        'FilePath': filepath
    }
    
    # Run all 4 algorithms
    ml_distribution = run_ml(profile)
    radius_distribution = run_radius(profile)
    mixed_distribution = run_mixed(profile)
    rd_distribution = run_rd(profile)
    
    # Calculate worst-case distortion for each
    if ml_distribution is not None:
        result['ML'] = get_worst_case_distortion(profile, ml_distribution)
    else:
        result['ML'] = None
    
    if radius_distribution is not None:
        result['RaDiUS'] = get_worst_case_distortion(profile, radius_distribution)
    else:
        result['RaDiUS'] = None
    
    if mixed_distribution is not None:
        result['Mixed'] = get_worst_case_distortion(profile, mixed_distribution)
    else:
        result['Mixed'] = None
    
    if rd_distribution is not None:
        result['RD'] = get_worst_case_distortion(profile, rd_distribution)
    else:
        result['RD'] = None
    
    return result


def main():
    """Main execution function"""
    print("="*70)
    print("Spotify Worst Case Empirical Analysis Experiment")
    print("="*70)
    
    # Load profiles (path relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spotify_dir = os.path.join(project_root, "spotifyday")
    
    print("\nLoading Spotify profiles...")
    profiles = load_spotify_profiles(spotify_dir)
    print(f"Loaded {len(profiles)} profiles")
    
    if len(profiles) == 0:
        print("ERROR: No profiles found in ./spotifyday/")
        print("Please download the Spotify dataset from PrefLib #00047")
        return
    
    # Determine number of workers
    n_workers = min(cpu_count(), len(profiles))
    print(f"Using {n_workers} worker processes")
    
    # Process in parallel with progress bar
    print("\nProcessing profiles...")
    start_time = time.time()
    
    results = []
    with Pool(processes=n_workers) as pool:
        # Use imap for progress tracking with tqdm
        with tqdm(total=len(profiles), desc="Processing", unit="profile", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for result in pool.imap(process_day, profiles):
                results.append(result)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
    print(f"  Average: {elapsed_time/len(profiles):.2f} seconds per profile")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV in outputs directory
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, "spotify_worst_case_empirical_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    for algo in ['ML', 'RaDiUS', 'Mixed', 'RD']:
        if algo in df.columns:
            values = df[algo].dropna()
            if len(values) > 0:
                print(f"{algo:8s}: Max={values.max():.4f}, Mean={values.mean():.4f}, Min={values.min():.4f}")
    
    return df


if __name__ == "__main__":
    df = main()

