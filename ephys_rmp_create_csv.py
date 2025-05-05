import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import seaborn as sns

# this script extracts the resting membrane potential (RMP) from ABF files and saves the data to a CSV file
# NEED TO MANUALLY ADD COLUMN "categories" TO THE CSV FILES with "baseline", "NA", "wash" 

def extract_rmp(file_path):
    abf = pyabf.ABF(file_path)
    # Skip files with fewer than 5 sweeps
    if len(abf.sweepList) < 6:
        print(f"Skipping {file_path}: fewer than 5 sweeps")
        return None
    abf.setSweep(5)  # Use sweep 5 (adjust if needed)
    rmp = np.mean(abf.sweepY[:int(0.1 * abf.dataRate)])  # First 100 ms
    return rmp

# Specify the folder containing the ABF files
folder_path = "/Volumes/joeschgrp/Group Members/Rima/Ephys_NE/DATA/20.02.2025 M1"

# Initialize a list to store file information
file_data = []

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".abf"):  # Check for ABF files
        file_path = os.path.join(folder_path, file_name)
        rmp = extract_rmp(file_path)
        if rmp is not None:  # Only add files with valid RMP values
            file_data.append({"file_name": file_name, "rmp": rmp})

# Convert the data to a DataFrame
df = pd.DataFrame(file_data)


csv_file = f"{os.path.basename(folder_path)}_rmp.csv"
df.to_csv(csv_file, index=False)
print(f"RMP data saved to {csv_file}")
