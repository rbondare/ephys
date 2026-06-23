"""
Script to analyze how firing rate changes across different red pulse timings
for baseline (control) vs Noradrenaline recordings.

Paradigm:
- Red pulse (baseline) at 5s
- Blue light 6-8s (triggers NA release)
- Red pulses at 8.1s, 10.1s, 12.1s, 14.1s, 16.1s (post_stim_1 through post_stim_5)

For each cell, calculates:
1. Average firing rate across all sweeps for each red pulse timing
2. Converts counts to firing rate (Hz) using 0.8s time window
3. Averages across cells and calculates SEM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Settings
EXCEL_PATH = "/Users/rbondare/ephys/results/double_opto_database_V2.xlsx"
SHEET_NAME = "Noradrenaline"
TIME_WINDOW = 0.8  # seconds, time window over which peaks were counted

# Column mapping: pulse timing to column name
PULSE_COLUMNS = {
    "Baseline\n(5s)": "baseline_avg",
    "Post-stim 1\n(+3.1s)": "post_stim_1_avg",
    "Post-stim 2\n(+5.1s)": "post_stim_2_avg",
    "Post-stim 3\n(+7.1s)": "post_stim_3_avg",
    "Post-stim 4\n(+9.1s)": "post_stim_4_avg",
    "Post-stim 5\n(+11.1s)": "post_stim_5_avg",
}

# Load data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=1)

# Separate baseline and NA conditions
baseline_df = df[df['condition'].str.lower() == 'baseline'].copy()
na_df = df[df['condition'].str.lower() == 'noradrenaline'].copy()

print(f"Loaded {len(baseline_df)} baseline cells and {len(na_df)} NA cells")

# Function to process condition data
def process_condition(condition_df, condition_name):
    """
    Convert peak counts to firing rates and organize by pulse timing.

    Returns:
    --------
    pulse_data : dict
        Keys are pulse labels, values are arrays of firing rates for each cell
    """
    pulse_data = {}

    for pulse_label, column_name in PULSE_COLUMNS.items():
        # Convert counts to firing rate (Hz)
        firing_rates = condition_df[column_name].values / TIME_WINDOW
        # Remove NaN values
        firing_rates = firing_rates[~np.isnan(firing_rates)]
        pulse_data[pulse_label] = firing_rates

    return pulse_data

# Process both conditions
baseline_data = process_condition(baseline_df, "Baseline")
na_data = process_condition(na_df, "Noradrenaline")

# Calculate statistics
def calculate_stats(pulse_data):
    """Calculate mean and SEM for each pulse."""
    means = []
    sems = []
    pulse_labels = []

    for label, firing_rates in pulse_data.items():
        means.append(np.mean(firing_rates))
        sems.append(stats.sem(firing_rates))
        pulse_labels.append(label)

    return pulse_labels, np.array(means), np.array(sems)

baseline_labels, baseline_means, baseline_sems = calculate_stats(baseline_data)
na_labels, na_means, na_sems = calculate_stats(na_data)

# Print summary statistics
print("\n" + "="*70)
print("BASELINE RECORDINGS (Control)")
print("="*70)
for label, mean, sem in zip(baseline_labels, baseline_means, baseline_sems):
    print(f"{label:25s}: {mean:6.2f} ± {sem:6.2f} Hz  (n={len(baseline_data[label])} cells)")

print("\n" + "="*70)
print("NORADRENALINE RECORDINGS")
print("="*70)
for label, mean, sem in zip(na_labels, na_means, na_sems):
    print(f"{label:25s}: {mean:6.2f} ± {sem:6.2f} Hz  (n={len(na_data[label])} cells)")

# Plot: Firing rate vs pulse timing
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(baseline_labels))
width = 0.35

# Plot bars with error bars
ax.bar(x_pos - width/2, baseline_means, width, label='Control (baseline only)',
       color='#908E8E', alpha=0.8, capsize=5)
ax.errorbar(x_pos - width/2, baseline_means, yerr=baseline_sems, fmt='none',
            color='black', capsize=5, linewidth=2, elinewidth=2)

ax.bar(x_pos + width/2, na_means, width, label='Noradrenaline',
       color='#EB6F6F', alpha=0.8, capsize=5)
ax.errorbar(x_pos + width/2, na_means, yerr=na_sems, fmt='none',
            color='black', capsize=5, linewidth=2, elinewidth=2)

# Labels and formatting
ax.set_xlabel('Red pulse timing', fontsize=13, fontweight='bold')
ax.set_ylabel('Firing rate (Hz)', fontsize=13, fontweight='bold')
ax.set_title('Firing Rate Across Red Pulse Timings:\nControl vs Noradrenaline',
             fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(baseline_labels, fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/rbondare/ephys-1/figures/na_timing_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot: Line plot showing response profile
fig, ax = plt.subplots(figsize=(10, 6))

x_pos_plot = np.arange(len(baseline_labels))

ax.plot(x_pos_plot, baseline_means, 'o-', linewidth=2.5, markersize=10,
        label='Control (baseline only)', color='#908E8E', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#908E8E')
ax.fill_between(x_pos_plot,
                baseline_means - baseline_sems,
                baseline_means + baseline_sems,
                alpha=0.2, color='#908E8E')

ax.plot(x_pos_plot, na_means, 's-', linewidth=2.5, markersize=10,
        label='Noradrenaline', color='#EB6F6F', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#EB6F6F')
ax.fill_between(x_pos_plot,
                na_means - na_sems,
                na_means + na_sems,
                alpha=0.2, color='#EB6F6F')

ax.set_xlabel('Red pulse timing', fontsize=13, fontweight='bold')
ax.set_ylabel('Firing rate (Hz)', fontsize=13, fontweight='bold')
ax.set_title('Firing Rate Response Profile:\nControl vs Noradrenaline',
             fontsize=15, fontweight='bold')
ax.set_xticks(x_pos_plot)
ax.set_xticklabels(baseline_labels, fontsize=11)
ax.legend(fontsize=12, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/rbondare/ephys-1/figures/na_timing_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical comparison: For each pulse, test if NA differs from baseline
print("\n" + "="*70)
print("STATISTICAL COMPARISON (paired t-tests)")
print("="*70)
for label in baseline_labels:
    baseline_vals = baseline_data[label]
    na_vals = na_data[label]

    # Only compare cells that have both measurements
    if len(baseline_vals) > 0 and len(na_vals) > 0:
        # Perform independent t-test
        t_stat, p_val = stats.ttest_ind(baseline_vals, na_vals)
        print(f"\n{label}")
        print(f"  Baseline: n={len(baseline_vals)}, mean={np.mean(baseline_vals):.2f} ± {stats.sem(baseline_vals):.2f} Hz")
        print(f"  NA:       n={len(na_vals)}, mean={np.mean(na_vals):.2f} ± {stats.sem(na_vals):.2f} Hz")
        print(f"  t-test: t={t_stat:6.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

print("\n" + "="*70)
