
import os
import numpy as np 
import pyabf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, bessel, filtfilt, savgol_filter 
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed

xlim = [0.48, 0.54]
ylim = [-80, 70]

def detect_spikes(file_path, cols):
    abf = pyabf.ABF(file_path)
    total_spikes = 0
    all_spikes = []
    spike_counts = []

 

    for sweep_index in range(abf.sweepCount):
        abf.setSweep(sweep_index)
        sweep_data = abf.sweepY

        # Calculate the threshold
        threshold = np.mean(sweep_data) + 20 * np.std(sweep_data)

        # Detect peaks
        peaks, _ = find_peaks(sweep_data, height=threshold)
        all_spikes.append(peaks)
        spike_counts.append(len(peaks))
        total_spikes += len(peaks)

    # Plot all sweeps with spikes detected
    num_sweeps = len(all_spikes)
    rows = (num_sweeps + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(8, rows * 1.5))
    fig.suptitle(f"Individual Sweeps - {file_path}")

    axs = axs.flatten()

    for i, spikes in enumerate(all_spikes):
        abf.setSweep(i)
        axs[i].plot(abf.sweepX, abf.sweepY, color='black', alpha=0.5)
        axs[i].scatter(abf.sweepX[spikes], abf.sweepY[spikes], color='red')
        axs[i].set_title(f"Sweep {i + 1}")
        axs[i].set_xlim(xlim) # comment out to see full trace 
        axs[i].set_ylim(ylim)
    

    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return len(spike_counts), total_spikes  # Return total number of sweeps and spikes


detect_spikes("/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA/29.01.2025 M2/2025_01_29_0040.abf", 5) 