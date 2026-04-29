"""
Script to plot Bar Graph of SC Current-Clamp Responses to RGC optogenetic activation
Spike_detection_single.py is a script which produces Probability of Spike (normalised by number of sweeps)
Data is grouped into "ntsr1" "inhibitory" and "WT" 
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Consistent color palette for light colors
colors = ['lightgrey', 'lightcoral', 'lightblue']

# Function to plot data for a given group
def plot_group_data(group_name, group_data):
    # Convert to DataFrame
    df = pd.DataFrame(group_data)
    df_long = df.melt(var_name='Condition', value_name='Value')
    df_long['Subject'] = df_long.groupby('Condition').cumcount()

    # Get conditions
    conditions = df.columns.tolist()

    # Start plot
    plt.figure(figsize=(4, 6))

    # Plot each subject's line
    for i in range(len(df)):
        plt.plot(range(len(conditions)), df.iloc[i, :], color='lightgray', alpha=0.8, linewidth=1)

    # Plot individual dots
    for idx, condition in enumerate(conditions):
        y = df[condition]
        x = np.random.normal(idx, 0.04, size=len(y))  # Jitter
        plt.scatter(x, y, color=colors[idx], s=60, zorder=3)

        # Plot mean as thick black bar
        plt.hlines(np.mean(y), idx - 0.15, idx + 0.15, color='black', linewidth=2.5)

    # Formatting
    plt.xticks(range(len(conditions)), conditions, rotation=45)
    plt.ylabel("Probability of Spike per Sweep")
    plt.title(f"{group_name}", fontsize=12)
    plt.tight_layout()
    plt.show()

   # Save the plot as a PDF
    # plt.savefig(f"{group_name}_plot.pdf", format="pdf")
    # plt.close()  # Close the plot to avoid overlapping plots



# Data
# from 05.25 data only included 2025_05_15_0002 (need to include the rest of the data from new exp) 

data = {
    "inhibitory": {
        "baseline": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.95, 0.2, 1, 1, 1],
        "NA": [1, 1, 1, 1, 1, 1, 1, 1, 0.38, 0.66, 1, 0.87, 0.77, 1, 1],
        "wash": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0.49, 1, 0.93, 0.49, 0.45, 1]
    },
    "ntsr1": {
    "baseline": [0.82, 0.89, 0.93, 1, 0.85, 1, 0.8, 1, 0.95],  
    "NA": [0, 0.67, 0.4, 0, 0.78, 0, 0.76, 0.09, 0.4],
    "wash": [0.49, 0.16, 0.16, 0.74, 0.36, 0.24, 0.54, 0.27, 0.12] 
    },
    "WT": {
        "baseline": [1, 1, 1, 0.93, 0.6, 0.93, 1],
        "NA": [0.94, 0.62, 1, 0.81, 0, 0.3, 0.56],
        "wash": [0, 0.96, 1, 0.27, 0.22, 0.11, 0.94]
    }
}

# Plot all groups
for group_name, group_data in data.items():
    plot_group_data(group_name, group_data)