
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors 
import pyabf 
import seaborn as sns
import os



base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

abf1 = pyabf.ABF(os.path.join(base_dir, "30.01.2025 M3/2025_01_30_0013.abf"))
abf2 = pyabf.ABF(os.path.join(base_dir, "30.01.2025 M3/2025_01_30_0014.abf"))
abf3 = pyabf.ABF(os.path.join(base_dir, "30.01.2025 M3/2025_01_30_0015.abf"))
#abf4 = pyabf.ABF(os.path.join(base_dir, "30.01.2025 M3/2025_01_30_0009.abf"))
all_sweeps = []

opto_stim = 0.5 
baseline = 0
NA_start = 16
wash = 55

xlim = [0.48, 0.54]
#xlim = [0.45, 0.6] 

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


divnorm = colors.TwoSlopeNorm(vmin=-70., vcenter=0, vmax=40)

# Plot heatmap using imshow (better for large data)
plt.figure(figsize=(10, 6))
plt.imshow(sweep_data, aspect='auto', cmap='RdBu_r', norm = divnorm, origin='lower',
           extent=[abf1.sweepX[0], abf1.sweepX[-1], 0, num_sweeps])

# Labels and Titles
#plt.axvline(x=opto_stim, color='red', linewidth=2)
plt.axhline(y=NA_start, color='white', linewidth=2, label="NA start")
plt.axhline(y=wash, color='white', linewidth=2, label="wash")
plt.colorbar(label="Voltage (mV)")
plt.xlabel("Time (s)")
plt.ylabel("Sweep Number")
plt.gca().invert_yaxis()  # Invert the y-axis to make Sweep 0 appear at the top
plt.xlim(xlim)
plt.show()
