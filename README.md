## 1) Repository Scope

The repository analyzes electrophysiology ABF recordings for:
- spike detection and firing-related metrics,
- resting membrane potential (RMP),
- voltage-clamp traces and sIPSC-related plotting,
- waveform extraction/alignment,
- optogenetic stimulation analyses and heatmaps.

Data processing is split between Python scripts and Jupyter notebooks.

## 2) Environment

Use the Conda environment in environment.yml.

Core packages used across workflows:
- pyabf
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- statsmodels
- pingouin
- statannotations
- openpyxl
- jupyter

## 3) Core Data Contracts

### 3.1 Master table (main analysis table)

The common analysis table is an Excel file (all_peak_results_final.xlsx).

Minimum columns expected by most plotting notebooks/scripts:
- filepath: absolute path to ABF file
- ID: cell identifier
- genotype: genotype group
- condition: baseline / Noradrenaline / wash
- comment: optional free text (rows with comment == "exclude" are often dropped)

Frequently used computed columns:
- normalized_peaks
- selected_10sweeps
- rmp (or equivalent RMP column)

### 3.2 Condition vocabulary

Most plotting/statistics assumes exactly three within-cell conditions:
- baseline
- Noradrenaline
- wash

Normalize spelling/case before pivoting and stats.

## 4) Global Dependency Graph

ABF files are the raw source. Most workflows follow this shape:

1. Read ABF + metadata (filepath, ID, genotype, condition)
2. Compute metric (spikes, firing rate, RMP, waveform features)
3. Write intermediate table (Excel/CSV)
4. Pivot into wide format by condition
5. Plot grouped trajectories and run stats

Cross-file dependency highlights:
- ephys_functions.py is the central reusable module for spike counting and genotype plotting.
- ephys_plotting_by_group.ipynb imports and uses functions from ephys_functions.py.
- RMP notebooks in rmp/ are mostly self-contained but rely on the same master-table schema.
- Waveform scripts in waveform/ are standalone and ABF-driven.

## 5) Python Modules (Detailed)

## 5.1 ephys_functions.py (central library)

Purpose:
- Shared spike detection, counting, and grouped plotting helpers.

Defined functions and contracts:

1. detect_peaks(data_array, height_threshold=2, prominence_min=0.1, distance_min=5)
- Depends on: scipy.signal.find_peaks
- Input: 1D signal array and thresholds
- Output: peaks indices + peak properties
- Used by: count_peaks, count_peaks_selected_sweeps, notebook/local helpers

2. count_peaks(abf_file, condition, genotype, ID, time_window=None, peak_params=None, output_dir=...)
- Depends on: pyabf + detect_peaks
- Input: one ABF file + metadata
- Behavior: counts peaks over all sweeps (or within optional time_window)
- Output table rows include: file, condition, genotype, ID, total_peaks, total_sweeps, normalized_peaks
- Writes: all_peak_results.xlsx (if time_window is None) or first_peak_results.xlsx
- Returns: result dict for one file

3. count_peaks_selected_sweeps(abf_file, condition, genotype, ID, comment, time_window=None, peak_params=None, output_dir=...)
- Depends on: pyabf + detect_peaks
- Input: one ABF file + metadata
- Behavior: same concept as count_peaks but only last 10 sweeps
- Output table rows include: file, condition, genotype, ID, comment, total_peaks, total_sweeps, normalized_peaks
- Writes: selected_10sweeps_peak_results.xlsx
- Returns: result dict for one file

4. analyze_by_condition(file_info, column_to_use=..., results_csv_path=...)
- Input: metadata dataframe + results table path
- Behavior: groups by ID and maps each condition to selected metric
- Output: wide dataframe with ID/genotype and one column per condition

5. plot_by_genotype(results)
6. plot_by_genotype_jitter(results)
7. plot_by_genotype_log(results)
8. plot_by_genotype_ratio(results)
9. plot_by_genotype_figure(results, output_dir)
10. plot_by_genotype_stat(results, output_dir)
- Input contract: wide-format dataframe with columns ID, genotype, baseline, Noradrenaline, wash
- Output: matplotlib figures (and stats printouts in *_stat)
- Stats in plot_by_genotype_stat: repeated-measures ANOVA + pairwise posthoc tests

## 5.2 ephys_database.py

Purpose:
- Scan ABF files and compile protocol/metadata summary table.

Typical behavior:
- list ABF files in a directory,
- extract file-level metadata,
- print and optionally export a CSV summary.

Input:
- Folder containing .abf files.

Output:
- DataFrame/CSV (protocol metadata).

## 5.3 Spike-detection scripts

1. spike_detection_single.py
- Single-file visual debugging of peak detection across sweeps.
- Input: one hardcoded ABF path.
- Output: per-sweep plot with detected peaks + printed summary.

2. spike_detection_from_database.py
- Batch inspect files listed in a CSV database.
- Input: CSV with filepath column.
- Output: sequential plots and console summaries.

3. spike_detection_many.py
- Batch compute normalized peaks over files.
- Input: hardcoded folder path.
- Output: peak_detection.csv with total peaks/sweeps/normalized peaks.

4. epys_spike_detection.py
- Variant/duplicate-style script (filename typo: epys vs ephys).
- Functionally similar to older spike-detection patterns.

## 5.4 Visualization scripts

1. ephys_APs.py
- AP sweep plotting utility.
- Input: ABF path.
- Output: sweep subplot grids.

2. ephys_VC.py
- Voltage-clamp plotting and file-to-file comparison.
- Input: one or two ABF paths.
- Output: overlay traces + average trace plots.

3. ephys_heatmap_spikes.py
- Heatmap across concatenated sweeps/conditions aligned to opto stimulus.
- Input: hardcoded ABF paths (typically baseline/NA/wash triplet).
- Output: one 2D heatmap figure.

4. plotting_bar.py
- Grouped bar/strip/trajectory plotting for hardcoded summary arrays.
- Input: currently in-script data dicts.
- Output: comparative genotype-condition plots.



## 5.5 Waveform scripts (waveform/)

1. waveform.py
- Detect spikes and extract waveform snippets.
- Computes descriptive stats (mean/variance/skewness/kurtosis).
- Output: waveform cloud + mean waveform.

2. waveform_align.py
- Aligns waveforms on spike peak and computes stats.
- Output: aligned waveform plot + summary stats.

3. waveform_align_overlay.py
- Compares aligned waveforms across two windows (e.g., before vs during opto).
- Output: overlay comparison plot.

## 5.6 RMP scripts (rmp/)

1. rmp/ephys_rmp_create_csv.py
- Extracts RMP from ABFs (typically first 100 ms region per sweep), aggregates per file.
- Input: folder of ABFs.
- Output: *_rmp.csv table.

2. rmp/ephys_rmp_plotting.py
- Reads RMP CSV tables and produces grouped RMP plots.
- Input: CSV files from create_csv step.
- Output: cloud/box style RMP plots.

## 6) Notebook Workflows

## 6.1 ephys_plotting_by_group.ipynb

Purpose:
- Main grouped spike/firing plotting notebook tied to master Excel.

What it defines/does:
- Imports ephys_functions utilities.
- Defines process_last10_and_plot(...): computes last-10-sweep metrics per file and plots grouped results.
- Defines process_last10_and_add(...): computes average firing-rate-style metric over selected sweeps and writes selected_10sweeps into the input Excel.
- Later cells load master Excel, filter excluded rows, pivot by condition, then plot.

Important dependency notes:
- Plot cells currently pivot on firingRate_10sweeps_all in some places.
- Upstream computation cell writes selected_10sweeps.
- If firingRate_10sweeps_all is absent, pivot will fail with KeyError.

Inputs required:
- Excel with filepath, ID, genotype, condition, comment.
- Valid filepaths to ABFs.

Outputs produced:
- Updated Excel (selected_10sweeps column).
- Plot figures and optional statistical summaries.

Run order (recommended):
1. Setup/import cell.
2. Define helper functions.
3. Run compute/update cell that writes selected_10sweeps.
4. Load the same updated table for plotting.
5. Pivot on the actual produced metric column name.
6. Run plotting + stats cells.

## 6.2 rmp/rmp_plotting_by_genotypes_final.ipynb

Purpose:
- Main RMP analysis notebook (final version).

Typical flow:
1. Load master file.
2. Optional RMP append from ABFs (flag-controlled).
3. Build genotype/condition pivots.
4. Plot grouped RMP, delta-from-baseline, histogram/bar views.
5. Run ANOVA/posthoc comparisons.

Inputs required:
- Master file with filepath, ID, genotype, condition (comment optional).
- ABFs available if append step is enabled.

Outputs produced:
- Multiple RMP figures and statistics printouts.
- Potentially updated table including RMP values.

## 6.3 rmp/rmp_from_database.ipynb

Purpose:
- Alternative RMP extraction/plotting workflow from master database.

Inputs:
- Master file + ABFs.

Outputs:
- RMP summaries, pivots, plots, and stats.

## 6.4 Opto_analysis.ipynb

Purpose:
- Optogenetic firing-rate analysis (before/during/after stimulation windows).

Inputs:
- ABF files + per-file stimulation timing.

Outputs:
- Sweep-level and cell-level firing-rate summaries + plots.

## 6.5 Other notebooks

- ephys_spike_count.ipynb: spike-counting exploration notebook.
- plotting_firing_rate.ipynb: extended plotting/statistics experiments.
- plotting_sIPSCs.ipynb and sIPSCs_from_csv.ipynb: sIPSC analyses.
- ephys_VC.ipynb: voltage-clamp notebook counterpart.
- plotting_by_group_from_database.ipynb: currently empty placeholder.
- heatmap_plotting_figure.ipynb: currently empty placeholder.

## 7) End-to-End Run Orders (By Goal)

## 7.1 Spike-by-condition pipeline (group plots)

1. Prepare master Excel with required metadata columns.
2. Ensure filepath entries exist on current machine.
3. Run compute step (from ephys_plotting_by_group.ipynb or ephys_functions batch use) to create metric column.
4. Confirm the metric column name used in pivot matches produced column (selected_10sweeps or firingRate_10sweeps_all or normalized_peaks).
5. Build pivot table (ID/genotype x condition).
6. Run plotting and statistics.

## 7.2 RMP pipeline

1. Use rmp/rmp_plotting_by_genotypes_final.ipynb as primary.
2. Load master and verify columns.
3. Optionally append RMP from ABFs.
4. Build pivots and run plots/stats.
5. Save/export publication figures.

## 7.3 Opto heatmap + rate pipeline

1. Use Opto_analysis.ipynb for rate windows.
2. Use ephys_heatmap_spikes.py for condition heatmap.
3. Use waveform_align_overlay.py for waveform comparison if needed.

## 8) Known Confusion Points (And Fixes)

1. Metric-name mismatch in grouped plotting.
- Symptom: KeyError for firingRate_10sweeps_all.
- Cause: compute step writes selected_10sweeps but plot expects firingRate_10sweeps_all.
- Fix: standardize to one column name or map aliases before pivot.

2. Hardcoded paths across scripts.
- Symptom: file not found when switching machines/OS.
- Fix: centralize paths in one config cell/file and use OS-safe absolute paths.

3. Duplicate/legacy scripts.
- Symptom: multiple similar scripts with slightly different logic.
- Fix: treat ephys_functions.py as canonical reusable module; archive or deprecate duplicates.

## 9) Recommended Standardization (Practical)

1. Choose one canonical metric name for grouped plotting (recommended: firingRate_10sweeps_all or selected_10sweeps, but use one consistently).
2. Add a single config.py (or notebook setup cell) with all file paths and output dirs.
3. Add explicit schema validation helper before any pivot/stats step.
4. Keep one primary notebook per analysis type:
- spikes: ephys_plotting_by_group.ipynb
- RMP: rmp/rmp_plotting_by_genotypes_final.ipynb
- opto: Opto_analysis.ipynb
5. Mark placeholders/legacy files clearly as DRAFT or LEGACY.

## 10) Quick Index

Primary reusable module:
- ephys_functions.py

Primary spike notebook:
- ephys_plotting_by_group.ipynb

Primary RMP notebook:
- rmp/rmp_plotting_by_genotypes_final.ipynb

Supporting RMP scripts:
- rmp/ephys_rmp_create_csv.py
- rmp/ephys_rmp_plotting.py

Waveform utilities:
- waveform/waveform.py
- waveform/waveform_align.py
- waveform/waveform_align_overlay.py

Opto utilities:
- Opto_analysis.ipynb
- ephys_heatmap_spikes.py

Legacy/placeholder:
- OLD_plotting_spikes.py
- epys_spike_detection.py
- plotting_by_group_from_database.ipynb
- heatmap_plotting_figure.ipynb
