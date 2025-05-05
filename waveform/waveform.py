import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe
from scipy.signal import find_peaks
import os

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

# Load the ABF file
abf = pyabf.ABF(os.path.join(base_dir, "29.03.2025 M6/2025_03_29_0008.abf"))

# Define plot limits
xlim = [0.44, 0.49]  # Time range for plotting
ylim = [-80, 70]      # Voltage range for plotting

# Initialize an array to store all sweeps
all_sweeps = []

# Process each sweep
for sweep in abf.sweepList:
    abf.setSweep(sweep)
    mask = (abf.sweepX >= xlim[0]) & (abf.sweepX <= xlim[1])  # Filter data within xlim
    all_sweeps.append(abf.sweepY[mask])  # Append the filtered sweep data

# Convert the list of sweeps to a NumPy array
all_sweeps = np.array(all_sweeps)

# Detect spikes in each sweep and analyze their waveforms
spike_waveforms = []
for sweep in all_sweeps:
    # Detect spikes using find_peaks
    peaks, _ = find_peaks(sweep, height=0, prominence=0.1, distance=5)  # Adjust parameters as needed
    
    # Extract the waveform around each spike (e.g., ±5 samples)
    for peak in peaks:
        start = max(0, peak - 5)  # Ensure we don't go out of bounds
        end = min(len(sweep), peak + 5)
        spike_waveforms.append(sweep[start:end])

# Flatten spike waveforms into a single numerical array
# Flatten spike waveforms into a single numerical array
if len(spike_waveforms) > 0:
    concatenated_spikes = np.concatenate([np.array(waveform, dtype=np.float64) for waveform in spike_waveforms])
    stats = describe(concatenated_spikes)
    
    # Print descriptive statistics in a more accessible format
    print("\nDescriptive Statistics for the Detected Spikes:")
    print(f"Number of Observations: {stats.nobs}")
    print(f"Minimum Value: {stats.minmax[0]:.2f}")
    print(f"Maximum Value: {stats.minmax[1]:.2f}")
    print(f"Mean: {stats.mean:.2f}")
    print(f"Variance: {stats.variance:.2f}")
    print(f"Skewness: {stats.skewness:.2f}")
    print(f"Kurtosis: {stats.kurtosis:.2f}")
else:
    print("No spikes detected in the specified time range.")
# Plot all individual waveforms
plt.figure(figsize=(8, 4))
for sweep in all_sweeps:
    plt.plot(abf.sweepX[mask], sweep, color='gray', alpha=0.5, linewidth=0.8, label="Individual Waveform" if sweep is all_sweeps[0] else "")

# Plot the average waveform
average_waveform = np.mean(all_sweeps, axis=0)
plt.plot(abf.sweepX[mask], average_waveform, color='blue', linewidth=1.5, label="Average Waveform")

# Customize the plot
plt.xlabel("Time (s)", fontsize=10)
plt.ylabel("Voltage (mV)", fontsize=10)
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()