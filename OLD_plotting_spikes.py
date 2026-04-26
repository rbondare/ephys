import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 



inhibitory_all = {
    "baseline": [1, 1.2, 1.9, 1.0, 1.0, 1.95,1.0 ,0.95, 0.2, 0.1, 1.65 , 1],
    "NA": [1.53 , 1.06, 1.66,0.45, 1.0 ,1.87, 1.37, 1.0, 0.87, 0.8, 1.56,1],
    "wash": [1.28, 1, 1, 0, 1.0, 1.5, 1.0, 1.3, 0.93, 0.48, 1.6, 0.45] 
}

# only count the first opto (spike)
inhibitory = {
    "baseline": [1, 1, 1,1, 1, 1, 1, 1, 1, 1, 0.95, 0.2, 1, 1],
    "NA": [1, 1, 1, 1, 1, 1, 1,1, 0.38, 0.66, 1, 0.87, 0.77,1 ],
    "wash": [1, 1, 1, 1, 1, 1, 1,1, 0, 0.49, 1, 0.93, 0.49, 0.45]
} 

ntsr1 = {
    "baseline": [0.82, 0.89, 0.93],  
    "NA": [0, 0.67, 0.4],
    "wash": [0.48, 0.16, 0.16] 
}

WT_all = {
    "baseline": [1, 2.5, 1, 1.4, 0.6, 0.93, 1],
    "NA": [0.94, 0.62, 1, 1.4, 0, 0.3, 0.56],   
    "wash": [0, 0.95, 1, 0.2, 0.28, 0.11, 0.93]
}
                 

WT = {
    "baseline": [1, 1, 1, 1.93, 0.6, 0.93, 1],
    "NA": [0.94, 0.62, 1, 1.81, 0, 0.3, 0.56],   
    "wash": [0, 0.96, 1, 0.27, 0.22, 0.11, 0.94]
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

means_inhibitory = np.mean(values_inhibitory, axis=1)
errors_inhibitory = np.std(values_inhibitory, axis=1) / np.sqrt(values_inhibitory.shape[1])  


# Bar positions
x_positions_inhibitory = np.arange(len(categories_inhibitory))
x_positions_ntsr1 = np.arange(len(categories_ntsr1))
x_positions_WT = np.arange(len(categories_WT))


# Create plot
plt.figure(figsize=(8, 6))

# Plot bars
plt.bar(x_positions_inhibitory, means_inhibitory, yerr=errors_inhibitory, color=["gray", "red", "blue"], alpha=0.8, capsize=5, label='Mean ± SEM')

# Plot individual data points
for i in range(values_inhibitory.shape[1]):  # Loop through samples
    plt.plot(x_positions_inhibitory, values_inhibitory[:, i], marker='o', color='gray', linestyle='-', alpha=0.7, label='Individual data' if i == 0 else "")

# Customize plot
plt.xticks(x_positions_inhibitory, categories_inhibitory)
plt.ylabel("Probability of Spike per Sweep")
plt.tight_layout()

# Show plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data for inhibitory
inhibitory = {
    "baseline": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.95, 0.2, 1, 1],
    "NA": [1, 1, 1, 1, 1, 1, 1, 1, 0.38, 0.66, 1, 0.87, 0.77, 1],
    "wash": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0.49, 1, 0.93, 0.49, 0.45]
}

# Convert data to a format suitable for seaborn
categories_inhibitory = list(inhibitory.keys())
values_inhibitory = np.array([inhibitory[cat] for cat in categories_inhibitory])  # Convert to NumPy array

# Create a cloud plot (strip plot)
plt.figure(figsize=(8, 6))
sns.stripplot(data=values_inhibitory.T, palette=["gray", "red", "blue"], size=8, jitter=True)

# Add lines connecting the dots for each sample
for i in range(values_inhibitory.shape[1]):  # Loop through each sample
    plt.plot(np.arange(len(categories_inhibitory)), values_inhibitory[:, i], color="black", alpha=0.5, linewidth=0.8)

# Customize the plot
plt.xticks(ticks=np.arange(len(categories_inhibitory)), labels=categories_inhibitory)
plt.ylabel("Probability of Spike per Sweep")
plt.title("Cloud Plot with Lines for Inhibitory Data")
plt.tight_layout()

# Show the plot
plt.show()
#%%

data = {
    "inhibitory": {
        "baseline": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.95, 0.2, 1, 1],
        "NA": [1, 1, 1, 1, 1, 1, 1, 1, 0.38, 0.66, 1, 0.87, 0.77, 1],
        "wash": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0.49, 1, 0.93, 0.49, 0.45]
    },
    "ntsr1": {
        "baseline": [0.82, 0.89, 0.93],
        "NA": [0, 0.67, 0.4],
        "wash": [0.48, 0.16, 0.16]
    },
    "WT": {
        "baseline": [1, 1, 1, 0.93, 0.6, 0.93, 1],
        "NA": [0.94, 0.62, 1, 0.81, 0, 0.3, 0.56],
        "wash": [0, 0.96, 1, 0.27, 0.22, 0.11, 0.94]
    }
}

# Function to plot data for a given group
def plot_group_data(group_name, group_data):
    # Convert data to NumPy array
    categories = list(group_data.keys())
    values = np.array([group_data[cat] for cat in categories])

    # Calculate means and standard deviations
    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1)

    # Create a cloud plot (strip plot)
    plt.figure(figsize=(8, 6))
    sns.stripplot(data=values.T, palette=["gray", "red", "blue"], size=8, jitter=True)

    # Add lines connecting the dots for each sample
    for i in range(values.shape[1]):  # Loop through each sample
        plt.plot(np.arange(len(categories)), values[:, i], color="black", alpha=0.5, linewidth=0.8)

    # Add bars for means with standard deviation
    x_positions = np.arange(len(categories))
    plt.bar(x_positions, means, yerr=stds, color=["gray", "red", "blue"], alpha=0.5, capsize=5)

    # Customize the plot
    plt.xticks(ticks=x_positions, labels=categories)
    plt.ylabel("Probability of Spike per Sweep")
    plt.title(f"{group_name.capitalize()} Data")
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

for group_name, group_data in data.items():
    plot_group_data(group_name, group_data)


#plot_group_data("inhibitory", data["inhibitory"])
#plot_group_data("ntsr1", data["ntsr1"])
#plot_group_data("WT", data["WT"])
