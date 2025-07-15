
import pyabf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, bessel, filtfilt, savgol_filter 
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed

xlim = [0, 2]
ylim = [-80, 70]


base_dir = "Z:\\Group Members\\Rima\\Ephys_NE\\DATA\\"
threshold_file = os.path.join(base_dir, "29.01.2025 M2/2025_01_29_0041.abf")


def detect_spikes(file_path, cols):
    abf = pyabf.ABF(file_path)
    total_spikes = 0
    all_spikes = []
    spike_counts = []

    # Calculate the threshold for spike detection (it is based on a representive sweep)
    
    if threshold_file:
        threshold_abf = pyabf.ABF(threshold_file)
        threshold_abf.setSweep(0)
        threshold_data = threshold_abf.sweepY
        threshold = np.mean(threshold_data) + 20 * np.std(threshold_data)
    else:
        abf.setSweep(0)
        first_sweep_data = abf.sweepY
        threshold = np.mean(first_sweep_data) + 15 * np.std(first_sweep_data)    
    

    for sweep_index in range(abf.sweepCount):
        abf.setSweep(sweep_index)
        sweep_data = abf.sweepY

        # Calculate the threshold
        #threshold = np.mean(sweep_data) + 17 * np.std(sweep_data)

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

    # calculate normalized spikes per sweep 
    if num_sweeps >0:
        norm_spikes = total_spikes/num_sweeps
        print(f'Normalised spikes:"{norm_spikes:.2f} spikes/sweep')

    return file_path, norm_spikes  # Return total number of sweeps and spikes 