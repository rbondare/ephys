import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import describe
import os

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

# Load the ABF file
abf = pyabf.ABF(os.path.join(base_dir, "25.03.2025 M3/2025_03_25_0041.abf"))

# Define plot limits for two time windows
xlim_before = [0.1, 0.4]  # Time range before opto stimulation
xlim_during = [0.5, 0.512]  # Time range during opto stimulation
window_size = 60  
ylim = [-80, 70]  # Voltage range for plotting

# Function to process sweeps, detect peaks, and align waveforms
def process_and_align_waveforms(abf, xlim, window_size):
    aligned_waveforms = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        mask = (abf.sweepX >= xlim[0]) & (abf.sweepX <= xlim[1])  # Filter data within xlim
        sweep_data = abf.sweepY[mask]
        
        # Detect peaks in the sweep
        peaks, _ = find_peaks(sweep_data, height=0, prominence=0.1, distance=5)
        
        # Extract and align waveforms around each peak
        for peak in peaks:
            start = max(0, peak - window_size // 2)  # Ensure we don't go out of bounds
            end = min(len(sweep_data), peak + window_size // 2)
            waveform = sweep_data[start:end]
            
            # Only include waveforms of the correct length
            if len(waveform) == window_size:
                aligned_waveforms.append(waveform)
    
    # Convert to NumPy array and calculate the average waveform
    aligned_waveforms = np.array(aligned_waveforms)
    if len(aligned_waveforms) > 0:
        average_waveform = np.mean(aligned_waveforms, axis=0)
    else:
        average_waveform = np.zeros(window_size)  # Handle case with no detected peaks
    return average_waveform

# Process the "before" and "during" time windows
avg_waveform_before = process_and_align_waveforms(abf, xlim_before, window_size)
avg_waveform_during = process_and_align_waveforms(abf, xlim_during, window_size)

# Create a sample-based x-axis for the aligned waveforms
sample_axis = np.arange(-window_size // 2, window_size // 2)  # Samples relative to the peak

# Plot the overlay of average waveforms
plt.figure(figsize=(8, 4))
plt.plot(sample_axis, avg_waveform_before, label="Before Opto Stimulation", color="blue", linewidth=1.5)
plt.plot(sample_axis, avg_waveform_during, label="During Opto Stimulation", color="red", linewidth=1.5)

# Customize the plot
plt.title("Overlay of Average Waveforms Aligned to Peaks", fontsize=12)
plt.xlabel("Samples (Aligned to Peak)", fontsize=10)
plt.ylabel("Voltage (mV)", fontsize=10)
plt.ylim(ylim)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()