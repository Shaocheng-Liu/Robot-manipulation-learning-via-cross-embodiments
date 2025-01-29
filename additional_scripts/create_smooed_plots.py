import pandas as pd
import os

# only used for visualization in thesis

# Load the CSV file
chunks = os.listdir('additional_scripts/orignal_csv/')
name = 'button-press-topdown2'
for name in chunks:
    df = pd.read_csv(f'additional_scripts/orignal_csv/{name}') 

    # Calculate moving average
    window_size = 5  # Adjust the window size for smoothing
    df['Smoothed5Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 10  # Adjust the window size for smoothing
    df['Smoothed10Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 20  # Adjust the window size for smoothing
    df['Smoothed20Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 30  # Adjust the window size for smoothing
    df['Smoothed30Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 40  # Adjust the window size for smoothing
    df['Smoothed40Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Calculate moving average
    window_size = 50  # Adjust the window size for smoothing
    df['Smoothed50Reward'] = df['Value'].rolling(window=window_size, min_periods=1).mean()

    # Save the original and smoothed data to a new CSV
    df.to_csv(f'additional_scripts/smooted_csv/smoothed_{name}', index=False)
