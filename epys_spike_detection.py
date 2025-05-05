"""
Improved script to detect all peaks in each sweep of an ABF file within a specific time window.
"""

import pyabf
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import os
import glob

time_window = [0.45, 0.7]  


base_dir = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA/30.01.2025 M3"
abf_files = glob.glob(os.path.join(base_dir, "**/*.abf"), recursive=True)

# Function to detect peaks
def detect_all_peaks(data_array, height_threshold=2, prominence_min=0.1, distance_min=5):
    """
    Detects all peaks in the signal based on height, prominence, and distance.
    
    Parameters:
    - data_array: The signal array to analyze
    - height_threshold: Minimum height of peaks
    - prominence_min: Minimum prominence to be considered a peak
    - distance_min: Minimum distance between peaks (in samples)
    
    Returns:
    - peaks: Array of indices where peaks were detected
    - properties: Properties of the detected peaks
    """
    peaks, properties = find_peaks(
        data_array,
        height=height_threshold,
        prominence=prominence_min,
        distance=distance_min
    )
    return peaks, properties

# Initialize a DataFrame to store results
results = pd.DataFrame(columns=["File", "Total Peaks", "Total Sweeps", "Normalized Peaks"])

# Loop through each file
for abf_file in abf_files:
    abf = pyabf.ABF(abf_file)
    
    total_peaks = 0
    total_sweeps = len(abf.sweepList)  
    
    # Process each sweep
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        
        # Filter the data to the specified time window
        mask = (abf.sweepX >= time_window[0]) & (abf.sweepX <= time_window[1])
        filtered_x = abf.sweepX[mask]
        filtered_y = abf.sweepY[mask]
        
        # Use the existing detect_all_peaks function on the filtered data
        peaks, _ = detect_all_peaks(
            filtered_y,
            height_threshold=0,  # Lowered threshold to detect smaller peaks
            prominence_min=0.1,  # Reduced prominence to include less distinct peaks
            distance_min=5       # Minimum distance between peaks (adjust as needed)
        )
        
        # Add the number of peaks in this sweep to the total
        total_peaks += len(peaks)
    
    # Calculate normalized peaks (total peaks divided by total sweeps)
    normalized_peaks = total_peaks / total_sweeps if total_sweeps > 0 else 0
    
    # Add results for this file to the DataFrame
    results = pd.concat([results, pd.DataFrame([{
        "File": os.path.basename(abf_file),  # Add only the file name, not the full path
        "Total Peaks": total_peaks,
        "Total Sweeps": total_sweeps,
        "Normalized Peaks": normalized_peaks
    }])], ignore_index=True)

# Save the results to a CSV file
results.to_csv("peak_detection.csv", index=False)

print(results)
