
import numpy as np 
import matplotlib.pyplot as plt
import pyabf 
import seaborn as sns
import os

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

abf1 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M1/2025_05_12_0006.abf"))
abf2 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M1/2025_05_12_0007.abf"))
abf3 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M1/2025_05_12_0008.abf"))
#abf4 = pyabf.ABF(os.path.join(base_dir, "13.05.2025 M3/2025_05_13_0017.abf"))
#abf5 = pyabf.ABF(os.path.join(base_dir, "13.05.2025 M3/2025_05_13_0018.abf"))
all_sweeps = []

opto_stim = 0.5 
baseline = 0
NA_start = 20
wash = 60

#xlim = [0.48, 0.54]
xlim = [0.45, 0.6] #better for double spikes 


for abf in [abf1, abf2, abf3]:
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        all_sweeps.append(abf.sweepY)

# Convert list to 2D numpy array
sweep_data = np.array(all_sweeps)


# Create a 2D array to store the concatenated sweep data
num_sweeps = len(all_sweeps)
num_timepoints = len(abf1.sweepX)
sweep_data = np.zeros((num_sweeps, num_timepoints))

# Populate the 2D array with concatenated sweep data
for i, sweep in enumerate(all_sweeps):
    sweep_data[i, :] = sweep  # Each row is a sweep, each column is a time point

# Plot heatmap using imshow (better for large data)
plt.figure(figsize=(10, 6))
plt.imshow(sweep_data, aspect='auto', cmap='gray', origin='lower',
           extent=[abf1.sweepX[0], abf1.sweepX[-1], 0, num_sweeps])

# Labels and Titles
plt.axvline(x=opto_stim, color='red', linewidth=0.5)
plt.axhline(y=NA_start, color='white', linewidth=1, label="NA start")
plt.axhline(y=wash, color='white', linewidth=1, label="wash")
colorbar = plt.colorbar(label="Voltage (mV)")
colorbar.mappable.set_clim(-70, 40)  
plt.ylabel("Sweep Number")
plt.gca().invert_yaxis()  # Invert the y-axis to make Sweep 0 appear at the top
plt.xlim(xlim)
plt.show()
