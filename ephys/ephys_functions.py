"""
This script provides functions for plotting electrophysiology data from ABF files
including: peak detection, counting peaks across sweeps etc

"""

import pyabf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob 
from statannotations.Annotator import Annotator
from scipy.stats import f_oneway
from statsmodels.stats.anova import AnovaRM
import pingouin as pg


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


def windowed_mean_excised(y_window, sample_rate_hz, height_threshold=4, prominence_min=0.1,
                           distance_min=5, exclude_pre_ms=5.0, exclude_post_ms=15.0):
    """
    Mean of y_window with samples near each detected spike excised, so a window
    that contains action potentials can still yield a subthreshold ("resting")
    voltage estimate instead of being biased upward by spike waveforms.

    Detects spikes via detect_peaks, then for each detected peak masks out
    [peak - exclude_pre_ms, peak + exclude_post_ms] (asymmetric: a short
    pre-peak upstroke vs a longer post-peak afterhyperpolarization), clipped
    to the window's own bounds.

    Parameters:
    - y_window: 1D voltage array for a single time window of a single sweep
    - sample_rate_hz: sampling rate of y_window, used to convert ms to samples
    - height_threshold, prominence_min, distance_min: passed to detect_peaks
    - exclude_pre_ms, exclude_post_ms: exclusion half-widths around each peak

    Returns:
    - mean_value: mean of the non-excised samples, or NaN if every sample in
      the window was excised (e.g. continuous firing fills the window)
    - keep_mask: boolean array, True where a sample was kept
    - peak_indices: indices (local to y_window) where detect_peaks fired
    - n_excised: number of samples excised
    - n_total: total number of samples in y_window
    """
    y_window = np.asarray(y_window)
    n_total = len(y_window)
    keep_mask = np.ones(n_total, dtype=bool)

    if n_total == 0:
        return np.nan, keep_mask, np.array([], dtype=int), 0, 0

    peak_indices, _ = detect_peaks(
        y_window,
        height_threshold=height_threshold,
        prominence_min=prominence_min,
        distance_min=distance_min,
    )

    pre_samples = int(round(exclude_pre_ms / 1000 * sample_rate_hz))
    post_samples = int(round(exclude_post_ms / 1000 * sample_rate_hz))

    for p in peak_indices:
        lo = max(0, p - pre_samples)
        hi = min(n_total - 1, p + post_samples)
        keep_mask[lo:hi + 1] = False

    n_excised = int((~keep_mask).sum())
    if keep_mask.sum() == 0:
        return np.nan, keep_mask, peak_indices, n_excised, n_total

    mean_value = float(np.mean(y_window[keep_mask]))
    return mean_value, keep_mask, peak_indices, n_excised, n_total


def sweep_window_means_excised(x_sweep, y_sweep, stimulus_windows, sample_rate_hz,
                                height_threshold=4, prominence_min=0.1, distance_min=5,
                                exclude_pre_ms=5.0, exclude_post_ms=15.0):
    """
    Applies windowed_mean_excised to each (start, end, label, color) window in
    stimulus_windows, using the same x/y masking approach as
    count_and_plot_peaks_multi_window (opto/ephys_double_opto.ipynb).

    Returns:
    dict keyed by window label, each value a dict with keys
    "mean", "keep_mask", "peak_indices_global" (mapped back to indices into
    y_sweep, not y_window), "n_excised", "n_total".
    """
    results = {}
    for start, end, label, _color in stimulus_windows:
        window_mask = (x_sweep >= start) & (x_sweep <= end)
        y_window = y_sweep[window_mask]

        mean_value, keep_mask, peak_indices_local, n_excised, n_total = windowed_mean_excised(
            y_window, sample_rate_hz,
            height_threshold=height_threshold,
            prominence_min=prominence_min,
            distance_min=distance_min,
            exclude_pre_ms=exclude_pre_ms,
            exclude_post_ms=exclude_post_ms,
        )

        global_indices = np.where(window_mask)[0]
        peak_indices_global = global_indices[peak_indices_local] if n_total > 0 else np.array([], dtype=int)

        results[label] = {
            "mean": mean_value,
            "keep_mask": keep_mask,
            "peak_indices_global": peak_indices_global,
            "n_excised": n_excised,
            "n_total": n_total,
        }
    return results


def extract_excised_trajectory_from_abf(abf_file, stimulus_windows, sweep_indices=None,
                                         height_threshold=4, prominence_min=0.1, distance_min=5,
                                         exclude_pre_ms=5.0, exclude_post_ms=15.0):
    """
    Per-file spike-excised mean voltage for each window in stimulus_windows,
    averaged across sweeps (file-level summary of extract_excised_per_sweep).

    sweep_indices=None uses every sweep in the file. np.nanmean aggregation
    means one fully-excised sweep/window doesn't poison the file-level
    average; if every sweep is NaN for a given window, that window's
    trajectory value is NaN and a warning is printed (naming the file and
    window) rather than silently vanishing.

    Returns:
    dict keyed by window label -> mean voltage (float) or NaN.
    """
    per_sweep_df = extract_excised_per_sweep(
        abf_file, stimulus_windows, sweep_indices=sweep_indices,
        height_threshold=height_threshold, prominence_min=prominence_min, distance_min=distance_min,
        exclude_pre_ms=exclude_pre_ms, exclude_post_ms=exclude_post_ms,
    )

    trajectory = {}
    for label, group in per_sweep_df.groupby('window')['mean']:
        values_arr = group.values
        if np.all(np.isnan(values_arr)):
            print(f"Warning: all sweeps excised for window '{label}' in {os.path.basename(abf_file)}")
            trajectory[label] = np.nan
        else:
            trajectory[label] = float(np.nanmean(values_arr))

    return trajectory


def extract_excised_per_sweep(abf_file, stimulus_windows, sweep_indices=None,
                               height_threshold=4, prominence_min=0.1, distance_min=5,
                               exclude_pre_ms=5.0, exclude_post_ms=15.0):
    """
    Per-sweep spike-excised mean voltage for each window in stimulus_windows —
    the detail that extract_excised_trajectory_from_abf averages away. Useful
    for looking at within-file trends across repeated sweeps (e.g. whether a
    prolonged/repeated stimulus has a cumulative effect), not just a single
    file-level summary.

    sweep_indices=None uses every sweep in the file (no "last 10" cap, unlike
    extract_rmp_from_abf / extract_excised_trajectory_from_abf's default) so
    the full within-file time course is visible.

    Returns:
    DataFrame with one row per (sweep, window) pair: columns
    sweep, window, mean, n_excised, n_total.
    """
    abf = pyabf.ABF(abf_file)
    if sweep_indices is None:
        sweep_indices = abf.sweepList

    rows = []
    for sweep in sweep_indices:
        abf.setSweep(sweep)
        window_results = sweep_window_means_excised(
            abf.sweepX, abf.sweepY, stimulus_windows, abf.dataRate,
            height_threshold=height_threshold,
            prominence_min=prominence_min,
            distance_min=distance_min,
            exclude_pre_ms=exclude_pre_ms,
            exclude_post_ms=exclude_post_ms,
        )
        for label, result in window_results.items():
            rows.append({
                'sweep': sweep, 'window': label, 'mean': result['mean'],
                'n_excised': result['n_excised'], 'n_total': result['n_total'],
            })

    return pd.DataFrame(rows)


def count_peaks(abf_file, condition, genotype, ID, time_window=None, peak_params=None, output_dir="C:\\Users\\rbondare\\ephys\\results\\"):
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
    
    # Determine filename based on whether time_window is used
    if time_window is None:
        results_filename = "all_peak_results.xlsx"
    else:
        results_filename = "first_peak_results.xlsx"
    results_df_path = os.path.join(output_dir, results_filename)
    if os.path.exists(results_df_path):
        existing_df = pd.read_excel(results_df_path)
        new_df = pd.concat([existing_df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        new_df = pd.DataFrame([result_dict])
    new_df.to_excel(results_df_path, index=False)

    return result_dict



def count_peaks_selected_sweeps(abf_file, condition, genotype, ID, comment, time_window=None, peak_params=None, output_dir="C:\\Users\\rbondare\\ephys\\results\\"):
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
        "comment": comment,
        "total_peaks": total_peaks,
        "total_sweeps": total_sweeps,
        "normalized_peaks": normalized_peaks
    }
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df_path = os.path.join(output_dir, "selected_10sweeps_peak_results.xlsx")
    if os.path.exists(results_df_path):
        existing_df = pd.read_excel(results_df_path)
        new_df = pd.concat([existing_df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        new_df = pd.DataFrame([result_dict])
    new_df.to_excel(results_df_path, index=False)

    return result_dict

def analyze_by_condition(file_info, column_to_use="normalized_first_peak", results_csv_path="C:\\Users\\rbondare\\ephys\\results\\all_peak_results.xlsx"):
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
    
    results_df = pd.read_excel(results_csv_path)
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
            plt.scatter(range(len(conditions)), y, color=colors, s=60, zorder=3, edgecolors='black', linewidths=0.7)
        
        # Plot mean as thick black bar for each condition (ensure mean is on top by plotting after scatter)
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            plt.hlines(np.mean(y_vals), i - 0.15, i + 0.15, color='black', linewidth=2.5, zorder=10)
        
        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Probability of Spike per Sweep")
        plt.title(f"{genotype}", fontsize=12)
        #plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()


def plot_by_genotype_jitter(results):
    """
    Plot spike probability (or chosen metric) by condition for each genotype.
    Each cell is linked across conditions with a line. One figure per genotype.
    Points are jittered horizontally for better visualization.
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with columns: ID, genotype, baseline, Noradrenaline, wash
    """
    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    
    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue

        plt.figure(figsize=(4, 6))
        
        # Create jittered x positions for each condition
        jitter_amount = 0.15  # Amount of horizontal jitter
        np.random.seed(42)  # For reproducible jitter
        
        # Plot each cell as a line across conditions
        for idx, row in group.iterrows():
            y = [row[cond] for cond in conditions]
            # Create jittered x positions for this cell
            x_jittered = [i + np.random.uniform(-jitter_amount, jitter_amount) for i in range(len(conditions))]
            
            plt.plot(x_jittered, y, color='lightgray', alpha=0.8, linewidth=1)
            plt.scatter(x_jittered, y, color=colors, s=60, zorder=2, edgecolors='black', linewidths=1)
        
        # Plot mean as thick black bar for each condition (at exact x positions)
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            if len(y_vals) > 0:
                plt.hlines(np.mean(y_vals), i - 0.15, i + 0.15, color='black', linewidth=2.5)
        
        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Average Firing Rate (Hz)")
        plt.title(f"{genotype}", fontsize=12)
        plt.xlim(-0.5, len(conditions) - 0.5)  # Set x limits to show jittered points properly
        #plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()


def plot_by_genotype_log(results):

    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    
    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue

        plt.figure(figsize=(4, 6))
        
        # Create jittered x positions for each condition
        jitter_amount = 0.15  # Amount of horizontal jitter
        np.random.seed(42)  # For reproducible jitter
        
        # Plot each cell as a line across conditions
        for idx, row in group.iterrows():
            y = [row[cond] for cond in conditions]
            # Create jittered x positions for this cell
            x_jittered = [i + np.random.uniform(-jitter_amount, jitter_amount) for i in range(len(conditions))]
            
            plt.plot(x_jittered, y, color='lightgray', alpha=0.8, linewidth=1)
            plt.scatter(x_jittered, y, color=colors, s=60, zorder=2, edgecolors='black', linewidths=1)
        
        # Plot mean as thick black bar for each condition (at exact x positions)
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            if len(y_vals) > 0:
                plt.hlines(np.mean(y_vals), i - 0.15, i + 0.15, color='black', linewidth=2.5)
        
        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Average Firing Rate (Hz)")
        plt.yscale('symlog')
        plt.title(f"{genotype}", fontsize=12)
        plt.xlim(-0.5, len(conditions) - 0.5)  # Set x limits to show jittered points properly
        plt.ylim(1e-3, 1e1)
        plt.tight_layout()
        plt.show()

def plot_by_genotype_ratio(results):
    """
    Plot all conditions normalized to baseline (baseline = 1, others as fold change).
    Each cell is linked across conditions with a line. One figure per genotype.
    Points are jittered horizontally for better visualization.
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with columns: ID, genotype, baseline, Noradrenaline, wash
    """
    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    
    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue

        plt.figure(figsize=(4, 6))
        
        # Create jittered x positions for each condition
        jitter_amount = 0.15  # Amount of horizontal jitter
        np.random.seed(42)  # For reproducible jitter
        
        # Plot each cell as a line across conditions (normalized to baseline)
        for idx, row in group.iterrows():
            baseline = row['baseline']

            
            # Normalize all values to baseline (baseline becomes 1)
            y = [row[cond] / baseline for cond in conditions]
            # Create jittered x positions for this cell
            x_jittered = [i + np.random.uniform(-jitter_amount, jitter_amount) for i in range(len(conditions))]
            
            plt.plot(x_jittered, y, color='lightgray', alpha=0.8, linewidth=1)
            plt.scatter(x_jittered, y, color=colors, s=60, zorder=3, edgecolors='black', linewidths=0.7)
        
        # Plot mean normalized values for each condition
        for i, cond in enumerate(conditions):
            normalized_vals = []
            for idx, row in group.iterrows():
                baseline = row['baseline']
                normalized_vals.append(row[cond] / baseline)
            if normalized_vals:
                normalized_vals_no_nan = [val for val in normalized_vals if not np.isnan(val)]
                if normalized_vals_no_nan:
                    plt.hlines(np.mean(normalized_vals_no_nan), i - 0.15, i + 0.15, color='black', linewidth=2.5, zorder=10)

        plt.xticks(range(len(conditions)), ['baseline\n(normalized)', 'NA', 'wash'], rotation=0)
        plt.ylabel("Fold Change (normalized to baseline)")
        plt.title(f"{genotype}", fontsize=12)
        plt.xlim(-0.5, len(conditions) - 0.5)  # Set x limits to show jittered points properly
        
        
        plt.tight_layout()
        plt.show()


def plot_by_genotype_figure(results, output_dir):
    """
    results : DataFrame
        DataFrame with columns: ID, genotype, baseline, Noradrenaline, wash
    """
    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    colors_bar = ["#A2A0A0", "#EB6F6F",  "#89CFF0"]  # Colors for the mean lines

    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue

        plt.figure(figsize=(4.5, 6))
        jitter_amount = 0.15
        np.random.seed(42)
        ax = plt.gca()
        
        # Prepare boxplot data
        box_data = [group[cond].dropna().values for cond in conditions]
        
        # Plot boxplots with only face color, no edges or whiskers visible
        box = ax.boxplot(
            box_data,
            positions=range(len(conditions)),
            widths=0.4,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='none'),
            boxprops=dict(facecolor='none', edgecolor='none'),
            whiskerprops=dict(color='none'),
            capprops=dict(color='none')
        )
        # Set facecolor of boxes with your colors but no edge
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)  # light shading
        
        # Plot individual points and connect pairs with lines
        for idx, row in group.iterrows():
            y = [row[cond] for cond in conditions]
            x_jittered = [i + np.random.uniform(-jitter_amount, jitter_amount) 
                          for i in range(len(conditions))]
            
            # Connect points with a thin gray line, using jittered x positions
            plt.plot(x_jittered, y, color='gray', alpha=0.3, linewidth=1, zorder=1)
            
            # Scatter points with color per condition
            for i, (x, val) in enumerate(zip(x_jittered, y)):
                plt.scatter(x, val, color=colors_bar[i], alpha=0.6, s=60, zorder=2,
                            edgecolors='black', linewidths=0.1)

        # Draw thick horizontal median lines for each condition
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            if len(y_vals) > 0:
                plt.hlines(np.median(y_vals), i - 0.2, i + 0.2,
                           color=colors_bar[i], linewidth=3.5, zorder=3)
            

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Firing Rate (Hz)")
        plt.title(f"{genotype}")
        plt.xlim(-0.5, len(conditions) - 0.5)  # Set x limits to show jittered points properly
    

        # Sanitize genotype string for filename (remove/replace problematic characters)
        safe_genotype = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(genotype))
        #plt.savefig(os.path.join(output_dir, f"{safe_genotype}_spike_probability.pdf"))
        plt.show()

def add_significance(ax, pairs, pvalues, y_offset=0.1, h=0.05, fontsize=12):
        """
        Draw significance bars and stars.
        """
        ymax = ax.get_ylim()[1]
        start_y = ymax + y_offset * ymax

        for i, ((x1, x2), pval) in enumerate(zip(pairs, pvalues)):
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'

            y = start_y + i * h * ymax
            ax.plot([x1, x1, x2, x2], [y, y+h*ymax*0.05, y+h*ymax*0.05, y],
                    lw=1.5, c='k')
            ax.text((x1+x2)/2, y+h*ymax*0.05, stars,
                    ha='center', va='bottom', fontsize=fontsize)

def plot_by_genotype_stat(results, output_dir):
    """
    results : DataFrame
        DataFrame with columns: ID, genotype, baseline, Noradrenaline, wash
    """
    genotypes = results['genotype'].unique()
    conditions = ['baseline', 'Noradrenaline', 'wash']
    colors = ['lightgrey', 'lightcoral', 'lightblue']
    colors_bar = ["#A2A0A0", "#EB6F6F",  "#89CFF0"]  # Colors for the mean lines

    for genotype in genotypes:
        group = results[results['genotype'] == genotype]
        if group.empty:
            continue

        # ---------- Run repeated-measures ANOVA ----------
        # Melt to long format
        long_df = group.melt(
            id_vars=['ID', 'genotype'],
            value_vars=conditions,
            var_name='condition',
            value_name='firingRate_10sweeps_all'
        )

        try:
            aov = AnovaRM(
                data=long_df,
                depvar='firingRate_10sweeps_all',
                subject='ID',
                within=['condition']

            ).fit()
            print(f"\nRepeated-measures ANOVA for genotype {genotype}:")
            print(aov)

            # Post-hoc pairwise tests with Bonferroni correction
            posthoc = pg.pairwise_ttests(
                dv='firingRate_10sweeps_all',
                within='condition',
                subject='ID',
                data=long_df,
                padjust='bonf'
            )
            print("  Post-hoc (Bonferroni corrected):")
            for _, row in posthoc.iterrows():
                A, B, pval = row['A'], row['B'], row['p-corr']
                print(f"    {A} vs {B}: p = {pval:.4f}")
        except Exception as e:
            print(f"Error during ANOVA or post-hoc tests for genotype {genotype}: {e}")

        # ---------- Plotting ----------
        plt.figure(figsize=(4.5, 6))
        jitter_amount = 0.15
        np.random.seed(42)
        ax = plt.gca()
        
        # Plot individual points and connect pairs with lines
        for idx, row in group.iterrows():
            y = [row[cond] for cond in conditions]
            x_jittered = [i + np.random.uniform(-jitter_amount, jitter_amount) 
                          for i in range(len(conditions))]
            
            # Connect points with a thin gray line, using jittered x positions
            plt.plot(x_jittered, y, color='gray', alpha=0.3, linewidth=1, zorder=1)
            
            # Scatter points with color per condition
            for i, (x, val) in enumerate(zip(x_jittered, y)):
                plt.scatter(x, val, color=colors_bar[i], alpha=0.6, s=60, zorder=2,
                            edgecolors='black', linewidths=0.1)

        # Draw thick horizontal median lines for each condition
        for i, cond in enumerate(conditions):
            y_vals = group[cond].dropna().values
            if len(y_vals) > 0:
                plt.hlines(np.median(y_vals), i - 0.2, i + 0.2,
                           color=colors_bar[i], linewidth=3.5, zorder=3)

        if posthoc is not None:
            pairs = []
            pvals = []
            mapping = {cond: i for i, cond in enumerate(conditions)}
            for _, row in posthoc.iterrows():
                pairs.append((mapping[row['A']], mapping[row['B']]))
                pvals.append(row['p-corr'])
            add_significance(ax, pairs, pvals)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        plt.xticks(range(len(conditions)), ['baseline', 'NA', 'wash'], rotation=45)
        plt.ylabel("Firing Rate (Hz)")
       # plt.title(f"{genotype}")
        plt.xlim(-0.5, len(conditions) - 0.5)  # Set x limits to show jittered points properly
    
        # Save or show
        safe_genotype = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(genotype))
        plt.savefig(os.path.join(output_dir, f"{safe_genotype}_firingrate.svg"))
        plt.show()


def main(csv_file_path, output_dir="C:\\Users\\rbondare\\ephys\\results\\"):
    """
    Main function to analyze spike probability from ABF files.

    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file with file information
    output_dir : str, optional

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
        file_info = pd.read_excel(csv_file_path)
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
    skipped = len(file_info.index) != len(pd.read_excel(csv_file_path))
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
        results_path = os.path.join(output_dir, 'spike_probability_results.xlsx')
        results.to_excel(results_path, index=False)
        print(f"Results saved to: {results_path}")

    # Plot results
    print("Generating plots by genotype...")
    try:
        plot_by_genotype(results)
    except Exception as e:
        print(f"Error generating plots: {e}")

    return results


