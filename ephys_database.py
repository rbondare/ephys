import pyabf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import pyabf 
import seaborn as sns
import os
import math 

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each cell
pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines

def list_abf_files(folder_dir):
    """
    Lists all .abf files in the given directory
    """
    return [f for f in os.listdir(folder_dir) if f.endswith('.abf')]

def compile_protocol_info(folder_dir):
    protocol_info = []
    abf_files = list_abf_files(folder_dir) 

    for abf_file in abf_files:
        try:
            abf_path = os.path.join(folder_dir, abf_file)
            abf = pyabf.ABF(abf_path)
            
            # Extract metadata from abf file
            protocol_name = abf.protocol
            sweep_number = len(abf.sweepList)
            recording_duration_seconds = abf.dataLengthSec
            recording_duration_minutes = recording_duration_seconds / 60
            
            # Get all comments and tags
            tags = ", ".join(abf.tagComments) if abf.tagComments else ""
            comments = abf.abfFileComment if hasattr(abf, 'abfFileComment') else ""
            
            protocol_info.append({
                'File Name': abf_file,
                'Protocol': protocol_name,
                'Sweep Number': sweep_number,
                'Duration (s)': f"{recording_duration_seconds:.1f}",
                'Duration (min)': f"{recording_duration_minutes:.1f}",
                'Tags': tags,
                'Comments': comments
            })
            
        except Exception as e:
            print(f"Error processing {abf_file}: {str(e)}")
            continue

    # Create DataFrame and sort by filename
    df = pd.DataFrame(protocol_info)
    df = df.sort_values(by='File Name')
    
    return df

if __name__ == "__main__":
    # Example usage
    folder_dir = input("Enter the path to your ABF files folder: ")
    metadata_df = compile_protocol_info(folder_dir)
    
    # Print a blank line before the table
    print("\nABF Files Metadata:")
    print("=" * 100)  # Separator line
    print(metadata_df.to_string(index=False))  # Print without index
    print("=" * 100)  # Separator line
    
    # Optional: Save to CSV if desired
    save = input("\nDo you want to save this to a CSV file? (yes/no): ").lower()
    if save == 'yes':
        output_path = os.path.join(folder_dir, 'abf_metadata.csv')
        metadata_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
