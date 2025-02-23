"""
This script plots individual sweeps from an ABF file as subplots. 
This is used for visualizing spiking activity across multiple sweeps in optogentic experiments 


For stimulation at 80 ms, xlim = [0.03, 0.15] #old experiments 
For stimulation at 500 ms, xlim = [0.45, 0.6] #new experiments


"""

import pyabf 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pyabf 
import seaborn as sns
import os
import math 

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"

#abf = pyabf.ABF(os.path.join(base_dir, "30.01.2025 M3/2025_01_30_0021.abf"))
abf = pyabf.ABF(os.path.join(base_dir, "25.07.24 M1/2024_07_25_0008.abf"))

xlim = [0.45, 0.6] #for subplots of 500 ms opto stim 
#xlim = [0.03, 0.15] #for subplots of 80 ms opto stim 
ylim = [-80, 70]
num_sweeps = len(abf.sweepList)
cols = 5  # Number of columns for subplots
rows = int(np.ceil(num_sweeps / cols)) 

# Reduce the overall figure size while maintaining proportions
#fig, axs = plt.subplots(rows, cols, figsize=(7, 8), sharex=True, sharey=True) 
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 0.8), sharex=True, sharey=True)

fig.suptitle(f"Individual Sweeps - {abf.abfID}")

# Flatten axs array to handle both 1D and 2D cases
axs = axes.flatten()

for i, sweep in enumerate(abf.sweepList):
    abf.setSweep(sweep)
    axs[i].plot(abf.sweepX, abf.sweepY, color='black', linewidth=0.6)
    axs[i].set_title(f"Sweep {sweep}", fontsize=5)
    axs[i].set_xlim(xlim)
    axs[i].set_ylim(ylim)
    axs[i].tick_params(labelbottom=True, labelsize=4)

# Hide unused subplots
for j in range(num_sweeps, len(axs)):
    axs[j].axis("off")

plt.tight_layout(rect=[0.05, 0.03, 1, 0.98], h_pad=0.3, w_pad=0.2)

plt.show()


