import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe
from scipy.signal import find_peaks
import os

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

# Load the ABF file
abf = pyabf.ABF(os.path.join(base_dir, "25.03.2025 M3/2025_03_25_0041.abf"))

# Define plot limits
xlim = [0.5, 0.51]  # Time range for plotting
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

# Detect spikes in each sweep and align their waveforms
spike_waveforms = []
for sweep in all_sweeps:
    # Detect spikes using find_peaks
    peaks, _ = find_peaks(sweep, height=0, prominence=0.1, distance=5)  # Adjust parameters as needed
    
# Extract and align the waveform around each spike (e.g., ±20 samples)
    for peak in peaks:
        start = max(0, peak - 30)  # Ensure we don't go out of bounds
        end = min(len(sweep), peak + 40)
        spike = sweep[start:end]
    
        # Align the spike by shifting it so the peak is centered
        if len(spike) == 70: 
            spike_waveforms.append(spike)

# Convert spike waveforms to a NumPy array
spike_waveforms = np.array(spike_waveforms)

# Analyze the aligned spike waveforms using scipy.stats.describe
if len(spike_waveforms) > 0:
    concatenated_spikes = np.concatenate(spike_waveforms)  # Flatten all spike waveforms for analysis
    stats = describe(concatenated_spikes)
    print("\nDescriptive Statistics for the Aligned Spikes:")
    print(f"Number of Observations: {stats.nobs}")
    print(f"Minimum Value: {stats.minmax[0]:.2f}")
    print(f"Maximum Value: {stats.minmax[1]:.2f}")
    print(f"Mean: {stats.mean:.2f}")
    print(f"Variance: {stats.variance:.2f}")
    print(f"Skewness: {stats.skewness:.2f}")
    print(f"Kurtosis: {stats.kurtosis:.2f}")
else:
    print("No spikes detected in the specified time range.")

# Plot all aligned waveforms
plt.figure(figsize=(8, 4))
for spike in spike_waveforms:
    plt.plot(spike, color='gray', alpha=0.5, linewidth=0.8, label="Aligned Waveform" if spike is spike_waveforms[0] else "")

# Plot the average aligned waveform
average_aligned_waveform = np.mean(spike_waveforms, axis=0)
plt.plot(average_aligned_waveform, color='blue', linewidth=1.5, label="Average Aligned Waveform")

# Customize the plot
plt.title("Aligned Spike Waveforms", fontsize=12)
plt.xlabel("Samples (Aligned to Peak)", fontsize=10)
plt.ylabel("Voltage (mV)", fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()