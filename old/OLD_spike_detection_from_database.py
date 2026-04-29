# This script detects peaks (single file) IN A GIVEN TIME RANGE (can be changed to entire trace) and plots with the detected peaks
# it takes a database.csv as an input to read the ABF file paths and produces one plot at a time per file to see which peaks are detected 

# CAN ADJUST THIS TO DETECT LAST 10 SWEEPS??? 


import pyabf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob 

def detect_all_peaks(data_array, height_threshold=2, prominence_min=0.1, distance_min=5):
    peaks, properties = find_peaks(
        data_array,
        height=height_threshold,
        prominence=prominence_min,
        distance=distance_min
    )
    return peaks, properties

# Configure matplotlib for interactive mode
plt.ion()  # Turn on interactive mode

csv_file_path = '/Users/rbondare/ephys/metadata/ephys_all_data.csv'  
file_info = pd.read_csv(csv_file_path, delimiter=';')

#time_range = [0.5, 0.515] # this time range to detect only the FIRST SPIKE 
time_range = [0, 4] #for the entire duration of the sweep
xlim = [0.45, 1]

total_files = len(file_info)
print(f"Total files to process: {total_files}")

for file_idx, (_, row) in enumerate(file_info.iterrows(), 1):
    abf_file = row['filepath']
    print(f"\nProcessing file {file_idx}/{total_files}: {os.path.basename(abf_file)}")
    
    try:
        abf = pyabf.ABF(abf_file)   
        num_sweeps = len(abf.sweepList)
        cols = 6 # Number of columns for subplots
        rows = int(np.ceil(num_sweeps / cols))
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.0), sharex=True, sharey=True)
        axs = axes.flatten() if rows > 1 or cols > 1 else [axes]  # Handle single subplot case

        all_peak_counts = [] 

        for i, sweep in enumerate(abf.sweepList):
            abf.setSweep(sweep)
            mask = (abf.sweepX >= time_range[0]) & (abf.sweepX <= time_range[1])
            filtered_x = abf.sweepX[mask]
            filtered_y = abf.sweepY[mask]
        
            peaks, peak_properties = detect_all_peaks(
                filtered_y,
                height_threshold=0,  # Lowered threshold to detect smaller peaks
                prominence_min=0.1,  # Reduced prominence to include less distinct peaks
                distance_min=5       # Minimum distance between peaks (adjust as needed)
            )
        
            # Append the number of peaks detected in this sweep to the list
            all_peak_counts.append(len(peaks))
        
            # Plot the sweep
            axs[i].plot(abf.sweepX, abf.sweepY, color='black', label=f"Sweep {sweep}")
        
            # Mark all detected peaks within the time range
            if len(peaks) > 0:
                axs[i].scatter(filtered_x[peaks], filtered_y[peaks], color='red', s=30)
        
            axs[i].set_title(f"Sweep {sweep}: {len(peaks)} peaks", fontsize=8)
            #axs[i].set_xlim(xlim)

        # Hide unused subplots if the number of sweeps is less than rows * cols
        for j in range(len(abf.sweepList), len(axs)):
            axs[j].axis('off')

        # Adjust layout and add title
        plt.tight_layout()
        plt.suptitle(f"File {file_idx}/{total_files}: {os.path.basename(abf_file)}", fontsize=10)
        
        # Print summary statistics for this file
        print(f"Average peaks per sweep: {np.mean(all_peak_counts):.2f}")
        print(f"Total peaks detected across all sweeps: {sum(all_peak_counts)}")
        
        # Show the figure and wait for user input
        plt.show()
        
        # Wait for user input before proceeding
        user_input = input("Press Enter to continue to next file, or 'q' to quit: ")
        if user_input.lower() == 'q':
            plt.close(fig)
            print("Stopped by user.")
            break
            
        # Close the current figure before proceeding
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing file {abf_file}: {str(e)}")
        continue

print("\nProcessing complete!")
plt.ioff()  # Turn off interactive mode
