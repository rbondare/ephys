""" 
This script to plot currents (voltage clamp) from ABF files 

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

base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA/gad"
abf = pyabf.ABF(os.path.join(base_dir, "26.03.2025 M4/2025_03_26_0005.abf"))

# Initialize a list to store all sweeps
all_sweeps = []

# Plot individual traces
plt.figure(figsize=(10, 6))
for sweep in abf.sweepList:
    abf.setSweep(sweep)
    plt.plot(abf.sweepX, abf.sweepY, color='black', linewidth=0.6, alpha=0.6)
    all_sweeps.append(abf.sweepY)

# Calculate and plot the average trace
average_trace = np.mean(all_sweeps, axis=0)
plt.plot(abf.sweepX, average_trace, color='red', linewidth=2, label="Average Trace")

# Customize the plot
plt.title("Voltage Clamp Currents with Average Trace")
plt.xlabel("Time (s)")
plt.ylabel("Current (pA)")
plt.legend(loc="upper right")
plt.grid(True)

# Set x-axis limits if needed
xlim = [0.45, 0.6]  # Adjust as per your requirement
plt.xlim(xlim)

# Show the plot
plt.tight_layout()
plt.show()


#%%

""" 
Plots 2 traces on top of each other to compare before and after NA
""" 


# Define the base directory and ABF file paths
base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA"
abf_file_1 = os.path.join(base_dir, "25.03.2025 M3/2025_03_25_0024.abf")
abf_file_2 = os.path.join(base_dir, "25.03.2025 M3/2025_03_25_0028.abf")  # Second ABF file

# Load the first ABF file
abf1 = pyabf.ABF(abf_file_1)
all_sweeps_1 = []

# Plot individual traces for the first file
plt.figure(figsize=(10, 6))
for sweep in abf1.sweepList:
    abf1.setSweep(sweep)
    plt.plot(abf1.sweepX, abf1.sweepY, color='black', linewidth=0.6, alpha=0.6)
    all_sweeps_1.append(abf1.sweepY)

# Calculate and plot the average trace for the first file
average_trace_1 = np.mean(all_sweeps_1, axis=0)
plt.plot(abf1.sweepX, average_trace_1, color='red', linewidth=2, label="Average Trace (File 1)")

# Load the second ABF file
abf2 = pyabf.ABF(abf_file_2)
all_sweeps_2 = []

# Plot individual traces for the second file
for sweep in abf2.sweepList:
    abf2.setSweep(sweep)
    plt.plot(abf2.sweepX, abf2.sweepY, color='blue', linewidth=0.6, alpha=0.6)
    all_sweeps_2.append(abf2.sweepY)

# Calculate and plot the average trace for the second file
average_trace_2 = np.mean(all_sweeps_2, axis=0)
plt.plot(abf2.sweepX, average_trace_2, color='green', linewidth=2, label="Average Trace (File 2)")

# Customize the plot
plt.title("Voltage Clamp Currents with Average Traces")
plt.xlabel("Time (s)")
plt.ylabel("Current (pA)")
plt.legend(loc="upper right")
plt.grid(True)

# Set x-axis limits if needed
xlim = [0.45, 0.6]  # Adjust as per your requirement
plt.xlim(xlim)

# Show the plot
plt.tight_layout()
plt.show()





