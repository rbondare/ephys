import os
import numpy as np 
import pyabf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks, bessel, filtfilt
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed


#1. List all .abf files in the specified directory
def list_abf_files(folder_dir):
    if os.path.exists(folder_dir):
        abf_files = [f for f in os.listdir(folder_dir) if f.endswith('.abf')]
        return abf_files
    else:
        print("The directory does not exist.")
        return []
    
    
#2. Compile protocol information from .abf files in the specified directory
def compile_protocol_info(folder_dir):
    protocol_info = []
    abf_files = list_abf_files(folder_dir)

    for abf_file in abf_files:
        abf_path = os.path.join(folder_dir, abf_file)
        abf = pyabf.ABF(abf_path)
        protocol_name = abf.protocol
        sweep_number = len(abf.sweepList)
        recording_duration_seconds = abf.dataLengthSec
        recording_duration_minutes = recording_duration_seconds / 60  # Convert to minutes
        tags = ", ".join(abf.tagComments)
        protocol_info.append((abf_file, protocol_name, sweep_number, recording_duration_seconds, recording_duration_minutes, tags))

    df = pd.DataFrame(protocol_info, columns=['File Name', 'Protocol', 'Sweep Number', 'Duration (s)', 'Duration (min)', 'Tags'])
    sorted_df = df.sort_values(by='Protocol')
    return sorted_df

#3    Process all subdirectories within data_dir and create a DataFrame for each
# def process_all_folders(data_dir):

#     all_dataframes = {}
    
#     for folder_name in os.listdir(data_dir):
#         folder_path = os.path.join(data_dir, folder_name)
#         if os.path.isdir(folder_path):
#             print(f"Processing folder: {folder_path}")
#             sorted_files = compile_protocol_info(folder_path)
#             if sorted_files is not None and not sorted_files.empty:
#                 all_dataframes[folder_name] = sorted_files
#                 print(f"Data for {folder_name} processed and stored in variable.")
#             else:
#                 print(f"No data to store for {folder_name}.")
#         else:
#             print(f"{folder_path} is not a directory.")
    
#     return all_dataframes


def process_single_folder(folder_info):
    """
    Process a single folder and return its data
    """
    folder_path, folder_name = folder_info
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")
        sorted_files = compile_protocol_info(folder_path)
        if sorted_files is not None and not sorted_files.empty:
            print(f"Data for {folder_name} processed and stored in variable.")
            return folder_name, sorted_files
        else:
            print(f"No data to store for {folder_name}.")
            return folder_name, None
    else:
        print(f"{folder_path} is not a directory.")
        return folder_name, None

def process_all_folders(data_dir, max_workers=8):
    """
    Process all folders in parallel using ThreadPoolExecutor
    
    Args:
        data_dir (str): Path to the directory containing folders to process
        max_workers (int, optional): Maximum number of worker threads. 
            If None, will default to min(32, os.cpu_count() + 4)
    
    Returns:
        dict: Dictionary containing processed dataframes for each folder
    """
    all_dataframes = {}
    
    # Create a list of (folder_path, folder_name) tuples
    folder_list = [
        (os.path.join(data_dir, folder_name), folder_name)
        for folder_name in os.listdir(data_dir)
    ]
    
    # Process folders in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all folder processing tasks
        future_to_folder = {
            executor.submit(process_single_folder, folder_info): folder_info[1]
            for folder_info in folder_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_folder):
            folder_name, result = future.result()
            if result is not None:
                all_dataframes[folder_name] = result
    
    return all_dataframes

#%%

def detect_spikes_and_plot_sweeps(abf, abf_file, cols):
    total_spikes = 0
    all_spikes = []
    spike_counts = []

    # Determine global min and max for y-axis
    global_min_voltage = np.inf
    global_max_voltage = -np.inf
 

    for sweep_index in range(abf.sweepCount):
        abf.setSweep(sweep_index)
        sweep_data = abf.sweepY

        # Calculate the threshold
        threshold = np.mean(sweep_data) + 20 * np.std(sweep_data)

        # Detect peaks
        peaks, _ = find_peaks(sweep_data, height=threshold)
        all_spikes.append(peaks)
        spike_counts.append(len(peaks))
        total_spikes += len(peaks)

        # Update global min and max
        global_min_voltage = min(global_min_voltage, np.min(sweep_data))
        global_max_voltage = max(global_max_voltage, np.max(sweep_data))

    # Plot all sweeps with spikes detected
    num_sweeps = len(all_spikes)
    rows = (num_sweeps + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 2))
    fig.suptitle(f"Individual Sweeps - {abf_file}")

    axs = axs.flatten()

    for i, spikes in enumerate(all_spikes):
        abf.setSweep(i)
        axs[i].plot(abf.sweepX, abf.sweepY, color='black', alpha=0.5)
        axs[i].scatter(abf.sweepX[spikes], abf.sweepY[spikes], color='red')
        axs[i].set_title(f"Sweep {i + 1}")
        axs[i].set_xlabel(abf.sweepLabelX)
        axs[i].set_ylabel(abf.sweepLabelY)
        axs[i].set_ylim(global_min_voltage, global_max_voltage)

    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return len(spike_counts), total_spikes  # Return total number of sweeps and spikes

# Main function for folder and file selection with updated analysis

def extract_number(filename):
    # Extract the numeric part from the filename (assuming it's in the format 'YYYY_MM_DD_XXXX.abf')
    base_name = filename.split('.')[0]  # Remove the extension
    number_part = base_name.split('_')[-1]  # Get the last part after the last underscore
    return int(number_part)

def analyze_CC(data_dir):

    subdirectories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print("Available folders:")

    for i, folder_name in enumerate(subdirectories):
        print(f"{i + 1}: {folder_name}")

    # Prompt user to select a folder
    folder_index = int(input("Enter the number of the folder you want to process: "))
    if 0 <= folder_index < len(subdirectories):
        folder_name = subdirectories[folder_index]
        folder_path = os.path.join(data_dir, folder_name)
        print(f"Processing folder: {folder_path}")

        # List available .abf files in the selected folder
        abf_files = [f for f in os.listdir(folder_path) if f.endswith('.abf')]
        abf_files.sort(key=extract_number)

        if abf_files:
            print("Available .abf files:")
            for i, abf_file in enumerate(abf_files):
                print(f"{i}: {abf_file}")

            # Choose the first and second files for analysis
            file_index_1 = int(input("Enter the number of the first file you want to process: "))
            file_index_2 = int(input("Enter the number of the second file you want to process: "))

            if 0 <= file_index_1 < len(abf_files) and 0 <= file_index_2 < len(abf_files):
                file1 = os.path.join(folder_path, abf_files[file_index_1])
                file2 = os.path.join(folder_path, abf_files[file_index_2])

                # Analyze the first file
                abf1 = pyabf.ABF(file1)
                abf2 = pyabf.ABF(file2)
                total_sweeps_1, total_spikes_1 = detect_spikes_and_plot_sweeps(abf1, abf_files[file_index_1], 10)
                total_sweeps_2, total_spikes_2 = detect_spikes_and_plot_sweeps(abf2, abf_files[file_index_2], 10)

                # Normalization example
                if total_sweeps_1 > 0 and total_sweeps_2 > 0:
                    norm_spikes_1 = total_spikes_1 / total_sweeps_1
                    norm_spikes_2 = total_spikes_2 / total_sweeps_2
                    print(f'Normalized spikes (File 1): {norm_spikes_1:.2f} spikes/sweep')
                    print(f'Normalized spikes (File 2): {norm_spikes_2:.2f} spikes/sweep')
                else:
                    print("One of the files has no sweeps.")
                    
                plot_average_sweeps_with_error(abf1, abf2, abf_files[file_index_1], abf_files[file_index_2]) 
                plot_average_sweeps(abf1, abf2, abf_files[file_index_1], abf_files[file_index_2]) 
                plot_pca(abf1, abf2, abf_files[file_index_1], abf_files[file_index_2])
                


            else:
                print("Invalid file numbers.")
        else:
            print("No .abf files found in the selected folder.")
    else:
        print("Invalid folder number.")


def plot_average_sweeps_with_error(abf1, abf2, file1, file2):
    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create a function to plot sweeps for a given ABF object
    def plot_sweeps_with_error(abf, color, label):
        all_sweeps = []
        for sweep_index in range(abf.sweepCount):
            abf.setSweep(sweep_index)
            all_sweeps.append(abf.sweepY)

        # Convert to array for averaging
        all_sweeps = np.array(all_sweeps)

        # Calculate average sweep and standard deviation
        average_sweep = np.mean(all_sweeps, axis=0)
        std_sweep = np.std(all_sweeps, axis=0)

        # Plot all sweeps with specified color and alpha
        #for sweep in all_sweeps:
        #  plt.plot(abf.sweepX, sweep, color=color, alpha=0.3)

        # Plot average sweep with thicker line in the same color
        plt.plot(abf.sweepX, average_sweep, color=color, label=f'Average ({label})', linewidth=2)

        # Fill between for the error cloud (standard deviation)
        plt.fill_between(abf.sweepX, average_sweep - std_sweep, average_sweep + std_sweep, 
                         color=color, alpha=0.2)

    # Plot for both files with different colors
    plot_sweeps_with_error(abf1, 'black', os.path.basename(file1))
    plot_sweeps_with_error(abf2, 'red', os.path.basename(file2))

    plt.xlim(0.05, 1) 
    plt.title('All Sweeps with Average and Error Cloud')
    plt.xlabel(abf1.sweepLabelX)
    plt.ylabel(abf1.sweepLabelY)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def plot_average_sweeps(abf1, abf2, file1, file2):
    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create a function to plot sweeps for a given ABF object
    def plot_sweeps(abf, color, label):
        all_sweeps = []
        for sweep_index in range(abf.sweepCount):
            abf.setSweep(sweep_index)
            all_sweeps.append(abf.sweepY)

        # Convert to array for averaging
        all_sweeps = np.array(all_sweeps)

        # Calculate average sweep and standard deviation
 #      average_sweep = np.mean(all_sweeps, axis=0)
 #       std_sweep = np.std(all_sweeps, axis=0)

        # Plot all sweeps with specified color and alpha
        for sweep in all_sweeps:
            plt.plot(abf.sweepX, sweep, color=color, alpha=0.5, linewidth=0.3)
            
          
    # Plot for both files with different colors
    plot_sweeps(abf1, 'black', os.path.basename(file1))
    plot_sweeps(abf2, 'red', os.path.basename(file2))
    plt.axvline(0.08122, color='blue')


    plt.xlim(0.05, 2)
    plt.title('All Sweeps with Average and Error Cloud')
    plt.xlabel(abf1.sweepLabelX)
    plt.ylabel(abf1.sweepLabelY)
    plt.legend()
    plt.grid()
    plt.show()


from sklearn.decomposition import PCA

def plot_pca(abf1, abf2, file1, file2):
    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create a function to plot sweeps for a given ABF object
    def plot_pca_core(abf, color, label):
        all_sweeps = []
        for sweep_index in range(abf.sweepCount):
            abf.setSweep(sweep_index)
            all_sweeps.append(abf.sweepY)
    
        # Convert to array
        all_sweeps = np.array(all_sweeps)
    
        # Perform PCA
        pca = PCA(n_components=1)  # We want the first component
        all_sweeps = np.transpose(all_sweeps)
        pca_result = pca.fit_transform(all_sweeps)
        
        plt.plot(abf.sweepX, pca_result, color=color, label=label)

    # Plot for both files with different colors
    plot_pca_core(abf1, 'black', os.path.basename(file1))
    plot_pca_core(abf2, 'red', os.path.basename(file2))

    plt.xlim(0.05, 1)
    plt.title('PC1 of all sweeps')
    plt.xlabel(abf1.sweepLabelX)
    plt.ylabel(abf1.sweepLabelY)
    plt.legend()
    plt.grid()
    plt.show()




