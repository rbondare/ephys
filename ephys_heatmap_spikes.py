import numpy as np 
import matplotlib.pyplot as plt
import pyabf 
import seaborn as sns
import os
sns.set_context('poster')

base_dir = "Z:\\Group Members\\Rima\\Ephys_NE\\DATA\\gad"


# abf1 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M4/2025_05_12_0001.abf"))
# abf2 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M4/2025_05_12_0002.abf"))
# abf3 = pyabf.ABF(os.path.join(base_dir, "12.05.2025 M4/2025_05_12_0003.abf"))

abf1 = pyabf.ABF(os.path.join(base_dir, "24.03.2025 M1/2025_03_24_0002.abf"))
abf2 = pyabf.ABF(os.path.join(base_dir, "24.03.2025 M1/2025_03_24_0003.abf"))
abf3 = pyabf.ABF(os.path.join(base_dir, "24.03.2025 M1/2025_03_24_0004.abf"))

all_sweeps = []
file_boundaries = []  
current_sweep_count = 0

opto_stim = 0.5 
xlim_relative = [-0.02, 0.05]  # 50ms before to 100ms after stimulation

abf_files = [abf1, abf2, abf3]
file_labels = ['Baseline', 'NA start', 'Wash']

for i, abf in enumerate(abf_files):
    file_boundaries.append(current_sweep_count)  # Record start of this file
    
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        all_sweeps.append(abf.sweepY)
        current_sweep_count += 1

# Convert list to 2D numpy array
sweep_data = np.array(all_sweeps)

# Create a 2D array to store the concatenated sweep data
num_sweeps = len(all_sweeps)
num_timepoints = len(abf1.sweepX)
sweep_data = np.zeros((num_sweeps, num_timepoints))

# Populate the 2D array with concatenated sweep data
for i, sweep in enumerate(all_sweeps):
    sweep_data[i, :] = sweep  # Each row is a sweep, each column is a time point

# Shift time axis so stimulation occurs at t=0
time_relative = abf1.sweepX - opto_stim

plt.figure()
plt.imshow(sweep_data, aspect='auto', cmap='bwr', origin='lower',
           extent=[time_relative[0], time_relative[-1], 0, num_sweeps])
# grey cmap looks nice 
# Add vertical line at stimulation time (now at x=0)
#plt.axvline(x=0, color='red', linewidth=2, label='Opto stimulation')

# Add horizontal lines at file boundaries
for boundary in file_boundaries[1:]:
    plt.axhline(y=boundary, color='white', linewidth=1, zorder=1, alpha=0.5)

# Labels and Titles
colorbar = plt.colorbar(label="Voltage (mV)")
colorbar.mappable.set_clim(-70, 40)  
plt.ylabel("Sweep Number")
plt.gca().invert_yaxis()  # Invert the y-axis to make Sweep 0 appear at the top
plt.xlabel("Time relative to stimulation (s)")
plt.xlim(xlim_relative)  # Show relative to stimulation
plt.tight_layout()
plt.show()