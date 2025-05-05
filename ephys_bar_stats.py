import numpy as np
import matplotlib.pyplot as plt



inhibitory = {
    "baseline": [1, 1.2, 1.9, 1.0, 1.0, 1.95,1.0 ,0.95, 0.2, 0.1, 1.65 , 1],
    "NA": [1.53 , 1.06, 1.66,0.45, 1.0 ,1.87, 1.37, 1.0, 0.87, 0.8, 1.56,1],
    "wash": [1.28, 1, 1, 0, 1.0, 1.5, 1.0, 1.3, 0.93, 0.48, 1.6, 0.45] 
}

ntsr1 = {
    "baseline": [0.82, 0.8, 0.93],  
    "NA": [0, 0.6, 0.4],
    "wash": [0.51, 0.16, 0.16] 
}

WT = {
    "baseline": [1, 2.5, 1, 1.4, 0.6, 0.93, 1],
    "NA": [0.94, 0.62, 1, 1.4, 0, 0.3, 0.56],   
    "wash": [0, 0.95, 1, 0.2, 0.28, 0.11, 0.93]
}
                 

# Convert to numpy arrays for easier manipulation
categories_inhibitory = list(inhibitory.keys())
values_inhibitory = np.array([inhibitory[cat] for cat in categories_inhibitory])  

categories_ntsr1 = list(ntsr1.keys())
values_ntsr1 = np.array([ntsr1[cat] for cat in categories_ntsr1])  

categories_WT = list(WT.keys())
values_WT = np.array([WT[cat] for cat in categories_WT]) 

means_WT = np.mean(values_WT, axis=1)
errors_WT = np.std(values_WT, axis=1) / np.sqrt(values_WT.shape[1])  

means_ntsr1 = np.mean(values_ntsr1, axis=1)
errors_ntsr1 = np.std(values_ntsr1, axis=1) / np.sqrt(values_ntsr1.shape[1])  

means = np.mean(values_inhibitory, axis=1)
errors = np.std(values_inhibitory, axis=1) / np.sqrt(values_inhibitory.shape[1])  


# Bar positions
x_positions = np.arange(len(categories_inhibitory))
x_positions_ntsr1 = np.arange(len(categories_ntsr1))
x_positions_WT = np.arange(len(categories_WT))


# Create plot
plt.figure(figsize=(8, 6))

# Plot bars
plt.bar(x_positions_WT, means_WT, yerr=errors_WT, color=["gray", "red", "blue"], alpha=0.8, capsize=5, label='Mean ± SEM')

# Plot individual data points
for i in range(values_WT.shape[1]):  # Loop through samples
    plt.plot(x_positions_WT, values_WT[:, i], marker='o', color='gray', linestyle='-', alpha=0.7, label='Individual data' if i == 0 else "")

# Customize plot
plt.xticks(x_positions_WT, categories_WT)
plt.ylabel("Probability of Spike per Sweep")
plt.tight_layout()

# Show plot
plt.show()
