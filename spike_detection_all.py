# This script detects peaks (single file) IN A GIVEN TIME RANGE and plots with the detected peaks

import pyabf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob 

def detect_all_peaks(data_array, height_threshold=0, prominence_min=0.1, distance_min=1):
    peaks, properties = find_peaks(
        data_array,
        height=height_threshold,
        prominence=prominence_min,
        distance=distance_min
    )
    return peaks, properties

base_dir = "Z:\\Group Members\\Rima\\Ephys_NE\\DATA\\ntsr1"
abf = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M5\\2025_05_12_0024.abf"))


# Create subplots for each sweep
num_sweeps = len(abf.sweepList)
cols = 6 # Number of columns for subplots
rows = int(np.ceil(num_sweeps / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.0), sharex=True, sharey=True)
axs = axes.flatten()  # Flatten the 2D array of axes into a 1D array


all_peak_counts = [] 

# Process each sweep
for i, sweep in enumerate(abf.sweepList):
    abf.setSweep(sweep)

    
    # Detect all peaks within the filtered time range
    peaks, peak_properties = detect_all_peaks(
        abf.sweepY,
        height_threshold=0,  # Lowered threshold to detect smaller peaks
        prominence_min=0.1,  # Reduced prominence to include less distinct peaks
        distance_min=1       # Minimum distance between peaks (adjust as needed)
    )
    
    # Append the number of peaks detected in this sweep to the list
    all_peak_counts.append(len(peaks))
    
    # Plot the sweep
    axs[i].plot(abf.sweepX, abf.sweepY, color='black', label=f"Sweep {sweep}")
    
    # Mark all detected peaks within the time range
    if len(peaks) > 0:
        axs[i].scatter(abf.sweepX[peaks], abf.sweepY[peaks], color='red', s=30)

    axs[i].set_title(f"Sweep {sweep}: {len(peaks)} peaks", fontsize=8)

# Hide unused subplots if the number of sweeps is less than rows * cols
for j in range(len(abf.sweepList), len(axs)):
    axs[j].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# Print summary statistics
print(f"Average peaks per sweep: {np.mean(all_peak_counts):.2f}")
print(f"Total peaks detected across all sweeps: {sum(all_peak_counts)}")