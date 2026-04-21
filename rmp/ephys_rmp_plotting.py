import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import seaborn as sns

csv_folder_path = "/Users/rbondare/Scripts_ephys/rmp_ntsr1"  # Replace with your folder path

# Initialize a list to store data from all CSV files
ntsr1 = []

# Loop through all CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
for file_name in csv_files:
    file_path = os.path.join(csv_folder_path, file_name)
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path, encoding="ISO-8859-1", delimiter=";")
    df["mouse"] = os.path.splitext(file_name)[0]  # Add a column for the mouse ID (from the file name)
    ntsr1.append(df)

# Combine all CSV files into a single DataFrame
combined_df = pd.concat(ntsr1, ignore_index=True)

# Replace NaN values in the "categories" column with "NA"
if "categories" in combined_df.columns:
    combined_df["categories"] = combined_df["categories"].fillna("NA")
else:
    raise ValueError("The 'categories' column is missing in the combined DataFrame.")

# Check the unique categories
print("\nUnique categories in the combined DataFrame:")
print(combined_df["categories"].unique())

# Organize data by categories
categories = combined_df["categories"].unique()
rmp_data = {category: combined_df[combined_df["categories"] == category]["rmp"].tolist() for category in categories}

# Find the maximum length of the lists
max_length = max(len(rmp_data[cat]) for cat in rmp_data)

# Pad shorter lists with NaN to make them the same length
padded_rmp_data = {cat: rmp_data[cat] + [np.nan] * (max_length - len(rmp_data[cat])) for cat in rmp_data}

# Convert the padded data to a NumPy array
categories_rmp = list(padded_rmp_data.keys())
values_rmp = np.array([padded_rmp_data[cat] for cat in categories_rmp])

# Calculate means and standard deviations, ignoring NaN values
means_rmp = np.nanmean(values_rmp, axis=1)
stds_rmp = np.nanstd(values_rmp, axis=1)

# Create a cloud plot (strip plot)
plt.figure(figsize=(8, 6))
sns.stripplot(data=values_rmp.T, palette=["gray", "red", "blue"], size=8, jitter=True)

# Add lines connecting the dots for each sample
for i in range(values_rmp.shape[1]):  # Loop through each sample
    plt.plot(np.arange(len(categories_rmp)), values_rmp[:, i], color="black", alpha=0.5, linewidth=0.8)

# Add bars for means with standard deviation
x_positions = np.arange(len(categories_rmp))
plt.bar(x_positions, means_rmp, yerr=stds_rmp, color=["gray", "red", "blue"], alpha=0.5, capsize=5)

# Customize the plot
plt.xticks(ticks=x_positions, labels=categories_rmp)
plt.ylabel("Resting Membrane Potential (mV)")
plt.title("Change in Resting Membrane Potential")
plt.tight_layout()
plt.ylim(-70, -40)  # Set y-axis limits
plt.gca().invert_yaxis()  # Invert y-axis
plt.show()


# # Create a violin plot for the categories
# plt.figure(figsize=(8, 6))
# sns.violinplot(data=values_rmp.T, palette=["gray", "red", "blue"])
# plt.xticks(ticks=np.arange(len(categories_rmp)), labels=categories_rmp)
# plt.xlabel("Categories")
# plt.ylabel("Resting Membrane Potential (mV)")
# plt.title("Violin Plot of Resting Membrane Potential by Category")
# plt.tight_layout()
# plt.show()


# Create a box plot for the categories
plt.figure(figsize=(8, 6))

# Create the box plot without outliers and with lighter box colors
sns.boxplot(
    data=values_rmp.T,
    palette=["gray", "red", "blue"],
    width=0.6,
    showfliers=False,  # Remove outliers
    boxprops={"alpha": 0.5}  # Set transparency for the box
)

# Overlay individual data points as dots
sns.stripplot(data=values_rmp.T, palette=["gray", "red", "blue"], size=8, jitter=True, alpha=0.7)

# Customize the plot
plt.xticks(ticks=np.arange(len(categories_rmp)), labels=categories_rmp)
plt.xlabel("Categories")
plt.ylabel("Resting Membrane Potential (mV)")
plt.title("Inhibitory - rmp")
plt.tight_layout()

# Show the plot
plt.show()

# Check the unique categories
print("\nUnique categories in the combined DataFrame:")
print(combined_df["categories"].unique())

# Verify that each value is assigned to the correct category
for category in ["baseline", "NA", "wash"]:
    print(f"\nValues assigned to category '{category}':")
    print(combined_df[combined_df["categories"] == category][["file_name", "rmp", "categories"]])