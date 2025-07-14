"""
This script provides functions for plotting electrophysiology data from ABF files
including: peak detection, counting peaks across sweeps, 

"""

import pyabf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob 

def detect_peaks(data_array, height_threshold=2, prominence_min=0.1, distance_min=5):
    
    """
    Detects all peaks in the signal based on height, prominence, and distance.
    
    Parameters:
    - data_array: The signal array to analyze
    - height_threshold: Minimum height of peaks
    - prominence_min: Minimum prominence to be considered a peak
    - distance_min: Minimum distance between peaks (in samples)
    
    Returns:
    - peaks: Array of indices where peaks were detected
    - properties: Properties of the detected peaks
    """
    peaks, properties = find_peaks(
        data_array,
        height=height_threshold,
        prominence=prominence_min,
        distance=distance_min
    )
    return peaks, properties


def count_peaks(abf_file, condition, genotype, ID, comment, time_window=None, peak_params=None, output_dir="/Users/rbondare/ephys/results"):
    """
    Count peaks in all sweeps of an ABF file, optionally within a time window, and associate results with a condition.
    
    Parameters:
    -----------
    abf_file : str
        Path to the ABF file
    condition : str
        Condition associated with the file (e.g., 'baseline', 'Noradrenaline', 'wash')
    time_window : list, optional
        [start_time, end_time] in seconds to analyze, if None, use entire sweep
    peak_params : dict, optional
        Parameters for peak detection: height_threshold, prominence_min, distance_min
    output_dir : str, optional
        Directory to save the results CSV file
    
    Returns:
    --------
    dict
        Dictionary with peak count results for this file
    """

    if peak_params is None:
        peak_params = {
            'height_threshold': 0,
            'prominence_min': 0.1,
            'distance_min': 3
        }
    
    abf = pyabf.ABF(abf_file)
    total_peaks = 0
    total_sweeps = len(abf.sweepList)
    sweeps_with_peaks = 0
    
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        sweep_data = abf.sweepY if time_window is None else abf.sweepY[(abf.sweepX >= time_window[0]) & (abf.sweepX <= time_window[1])]
        peaks, _ = detect_peaks(sweep_data, **peak_params)
        sweeps_with_peaks += len(peaks) > 0
        total_peaks += len(peaks)
    
    normalized_peaks = total_peaks / total_sweeps if total_sweeps > 0 else 0

    result_dict = {
        "file": os.path.basename(abf_file),
        "condition": condition, 
        "genotype": genotype,
        "ID": ID,
        "total_peaks": total_peaks,
        "total_sweeps": total_sweeps,
        "normalized_peaks": normalized_peaks
    }
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df_path = os.path.join(output_dir, "all_peak_results.csv")
    pd.DataFrame([result_dict]).to_csv(results_df_path, mode='a', header=not os.path.exists(results_df_path), index=False)
    
    return result_dict



def count_peaks_selected_sweeps(abf_file, condition, genotype, ID, comment, time_window=None, peak_params=None, output_dir="/Users/rbondare/ephys/results"):
    """
    Count peaks in all sweeps of an ABF file, optionally within a time window, and associate results with a condition.
    
    Parameters:
    -----------
    abf_file : str
        Path to the ABF file
    condition : str
        Condition associated with the file (e.g., 'baseline', 'Noradrenaline', 'wash')
    time_window : list, optional
        [start_time, end_time] in seconds to analyze, if None, use entire sweep
    peak_params : dict, optional
        Parameters for peak detection: height_threshold, prominence_min, distance_min
    output_dir : str, optional
        Directory to save the results CSV file
    
    Returns:
    --------
    dict
        Dictionary with peak count results for this file
    """

    if peak_params is None:
        peak_params = {
            'height_threshold': 0,
            'prominence_min': 0.1,
            'distance_min': 3
        }
    
    abf = pyabf.ABF(abf_file)
    total_peaks = 0
    
    # Only use the last 10 sweeps (or all if fewer than 10)
    sweep_indices = abf.sweepList[-10:] if len(abf.sweepList) > 10 else abf.sweepList
    total_sweeps = len(sweep_indices)
    sweeps_with_peaks = 0

    for sweep in sweep_indices:
        abf.setSweep(sweep)
        sweep_data = abf.sweepY if time_window is None else abf.sweepY[(abf.sweepX >= time_window[0]) & (abf.sweepX <= time_window[1])]
        peaks, _ = detect_peaks(sweep_data, **peak_params)
        sweeps_with_peaks += len(peaks) > 0
        total_peaks += len(peaks)
    
    normalized_peaks = total_peaks / total_sweeps if total_sweeps > 0 else 0

    result_dict = {
        "file": os.path.basename(abf_file),
        "condition": condition, 
        "genotype": genotype,
        "ID": ID,
        "total_peaks": total_peaks,
        "total_sweeps": total_sweeps,
        "normalized_peaks": normalized_peaks
    }
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df_path = os.path.join(output_dir, "selected_sweeps_peak_results.csv")
    pd.DataFrame([result_dict]).to_csv(results_df_path, mode='a', header=not os.path.exists(results_df_path), index=False)
    
    return result_dict

def analyze_by_condition(file_info, column_to_use="normalized_first_peak", results_csv_path="/Users/rbondare/ephys/results/all_peak_results.csv"):
    """
    Analyze ABF files grouped by cell ID and condition, using a specific column from the results CSV file.
    
    Parameters:
    -----------
    file_info : DataFrame
        DataFrame containing file information with columns:
        filepath, filename, genotype, condition, cell ID
    column_to_use : str, optional
        Column name from the results CSV file to use for further processing
    results_csv_path : str, optional
        Path to the results CSV file
    
    Returns:
    --------
    DataFrame
        Results grouped by cell ID and condition for the chosen column
    """
    if results_csv_path is None or not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Results CSV file not found at {results_csv_path}")
    
    results_df = pd.read_csv(results_csv_path)
    grouped_results = []
    
    for cell_idx, cell_group in file_info.groupby('ID'):
        genotype = cell_group['genotype'].iloc[0]
        cell_results = {"ID": cell_idx, "genotype": genotype}
        
        for condition in ['baseline', 'Noradrenaline', 'wash']:  
            condition_files = cell_group[cell_group['condition'] == condition]
            if condition_files.empty:
                cell_results[condition] = None
                continue
            
            file_name = condition_files['filename'].iloc[0]  

            chosen_value = results_df.loc[(results_df['file'] == file_name) & (results_df['condition'] == condition), column_to_use].values
            cell_results[condition] = chosen_value[0] if len(chosen_value) > 0 else None
        
        grouped_results.append(cell_results)
    
    return pd.DataFrame(grouped_results)

def plot_by_genotype(results):
    """
    Plot spike probability (or chosen metric) by condition for each genotype.
    Each cell is linked across conditions with a line. One figure per genotype.
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with columns: ID, genotype, baseline, Noradrenaline, wash
    ylim : tuple, optional
        y-axis limits for the plot (default: (-0.05, 1.05))
    """
    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    
    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue
        
        plt.figure(figsize=(4, 6))
        # Plot each cell as a line across conditions
        for idx, row in group.iterrows():
            y = [row[cond] for cond in conditions]
            plt.plot(range(len(conditions)), y, color='lightgray', alpha=0.8, linewidth=1)
            plt.scatter(range(len(conditions)), y, color=colors, s=60, zorder=3)
        
        # Plot mean as thick black bar for each condition
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            if len(y_vals) > 0:
                plt.hlines(np.mean(y_vals), i - 0.15, i + 0.15, color='black', linewidth=2.5)
        
        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Probability of Spike per Sweep")
        plt.title(f"{genotype}", fontsize=12)
        #plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()


def main(csv_file_path, output_dir='/Users/rbondare/ephys/ephys_results'):
    """
    Main function to analyze spike probability from ABF files.

    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file with file information
    output_dir : str, optional
        Directory to save results (default is '/Users/rbondare/ephys/ephys_results')
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return None

    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the CSV file
    print(f"Loading data from: {csv_file_path}")
    try:
        file_info = pd.read_csv(csv_file_path, delimiter=';')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Debug: print column names and first few rows
    print("CSV columns:", list(file_info.columns))
    print("First 3 rows:\n", file_info.head(3))

    required_columns = ['ID', 'genotype', 'condition', 'filepath']
    missing_columns = [col for col in required_columns if col not in file_info.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None

    # Filter out rows with invalid file paths
    file_info = file_info[file_info['filepath'].apply(os.path.exists)]
    if file_info.empty:
        print("Error: No valid ABF file paths found in CSV.")
        return None

    # Count and warn about skipped files
    skipped = len(file_info.index) != len(pd.read_csv(csv_file_path, delimiter=';'))
    if skipped:
        print("Warning: Some files had invalid paths and were skipped.")

    # Analyze by condition
    print("Analyzing data by condition...")
    try:
        results = analyze_by_condition(file_info)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

    # Save results
    if output_dir:
        results_path = os.path.join(output_dir, 'spike_probability_results.csv')
        results.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

    # Plot results
    print("Generating plots by genotype...")
    try:
        plot_by_genotype(results)
    except Exception as e:
        print(f"Error generating plots: {e}")

    return results