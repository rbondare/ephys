import numpy as np
import matplotlib.pyplot as plt


# Example data
data = {
    "baseline": [0.67, 0.93, 1],
    "NA": [0.14, 0.33, 0.56],
    "wash": [0.28, 0.11, 0.94]
}

# Convert to numpy arrays for easier manipulation
categories = list(data.keys())
values = np.array([data[cat] for cat in categories])  # Shape: (3, n_samples)

# Calculate means and standard errors
means = np.mean(values, axis=1)
errors = np.std(values, axis=1) / np.sqrt(values.shape[1])  # Standard error

# Bar positions
x_positions = np.arange(len(categories))

# Create plot
plt.figure(figsize=(8, 6))

# Plot bars
plt.bar(x_positions, means, yerr=errors, color=["gray", "red", "blue"], alpha=0.8, capsize=5, label='Mean ± SEM')

# Plot individual data points
for i in range(values.shape[1]):  # Loop through samples
    plt.plot(x_positions, values[:, i], marker='o', color='gray', linestyle='-', alpha=0.7, label='Individual data' if i == 0 else "")

# Customize plot
plt.xticks(x_positions, categories)
plt.ylabel("Probability of Spike per Sweep")
plt.tight_layout()

# Show plot
plt.show()
