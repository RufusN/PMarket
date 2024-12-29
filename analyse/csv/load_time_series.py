import pandas as pd
import glob
import os

def market_data(directory):
    """
    Load CSV files from a specified directory and merge them into a single DataFrame based on timestamp.
    """
    # Use glob to find all CSV files in the directory
    file_pattern = os.path.join(directory, "*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No CSV files found in the specified directory.")
        return pd.DataFrame()
    
    dataframes = []
    
    for file in all_files:
        # Load each CSV file into a DataFrame
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Convert timestamps
        df.set_index('timestamp', inplace=True)
        
        # Extract a meaningful name from the file (e.g., market ID + market name)
        market_name = os.path.basename(file).replace(".csv", "")
        df.rename(columns={'price': market_name}, inplace=True)  # Rename price column with the market name
        
        dataframes.append(df[market_name])  # Keep only the price column, with market name as column name
    
    # Combine all DataFrames into one, aligning on timestamps
    combined_df = pd.concat(dataframes, axis=1)
    
    return combined_df

def save_matrix_to_csv(data, output_path):
    """
    Save the combined DataFrame as a CSV matrix.
    """
    try:
        data.to_csv(output_path, index=True)
        print(f"Matrix saved to {output_path}")
    except IOError as e:
        print(f"Error saving matrix to CSV: {e}")

if __name__ == "__main__":
    directory = "/Users/ru/Polymarket/time_series"  # Adjust this to your directory
    output_path = "combined_matrix.csv"  # Path to save the matrix
    
    # Load the market data
    print("Loading market data...")
    combined_data = market_data(directory)
    
    if not combined_data.empty:
        print("Data Loaded. Head of the combined DataFrame:")
        print(combined_data.head())
        
        # Save the combined matrix to CSV
        save_matrix_to_csv(combined_data, output_path)
    else:
        print("No data was loaded. Ensure the directory contains valid CSV files.")
