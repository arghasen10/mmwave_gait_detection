import pandas as pd
import glob
import os

# Path to the folder containing the CSV files
folder_path = 'new_csvs'  # Replace with the path to your folder
output_filename = 'annotation.csv'

# Use glob to find all CSV files in the folder
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# List to hold dataframes
dfs = []

# Iterate over all files and read them
for filename in all_files:
    df = pd.read_csv(filename, encoding='utf-8-sig')  # Read each file
    dfs.append(df)  # Append dataframe to the list

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined dataframe to a single CSV file
combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Combined file saved as {output_filename}")
