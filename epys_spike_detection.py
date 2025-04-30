"""
Improved script to detect all peaks in each sweep of an ABF file.
"""

import pyabf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the base directory and ABF file path
base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"
abf = pyabf.ABF(os.path.join(base_dir, "24.03.2025 M1/2025_03_24_0007.abf"))

# Function to detect peaks
def detect_all_peaks(data_array, height_threshold=2, prominence_min=0.1, distance_min=5):
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

# Time window to focus on (in seconds)
xlim = [0.45, 0.7]

# Create subplots for each sweep
num_sweeps = len(abf.sweepList)
cols = 5  # Number of columns for subplots
rows = int(np.ceil(num_sweeps / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.0), sharex=True, sharey=True)
axs = axes.flatten()  # Flatten the 2D array of axes into a 1D array

# List to store all peak counts
all_peak_counts = []

# Process each sweep
for i, sweep in enumerate(abf.sweepList):
    abf.setSweep(sweep)
    
    # Detect all peaks
    peaks, peak_properties = detect_all_peaks(
        abf.sweepY,
        height_threshold=0,  # Lowered threshold to detect smaller peaks
        prominence_min=0.1,  # Reduced prominence to include less distinct peaks
        distance_min=5       # Minimum distance between peaks (adjust as needed)
    )
    
    all_peak_counts.append(len(peaks))
    
    # Plot the sweep
    axs[i].plot(abf.sweepX, abf.sweepY, color='black', label=f"Sweep {sweep}")
    
    # Mark all detected peaks
    if len(peaks) > 0:
        axs[i].scatter(abf.sweepX[peaks], abf.sweepY[peaks], color='red', s=30)
    
    axs[i].set_xlim(xlim)  # Set x-axis limits for each subplot
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