import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import ast
import json #for robustness in deserializing

# **Helper Functions**

def safe_parse(value):
    """Safely parse a value into a list."""
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (list, np.ndarray)):
            return value
        else:
            return None
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Malformed entry: {value} -> Error: {e}")
        return None

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data."""
    nyquist = 1 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)

# **Load and Clean Data**
eeg_data = pd.read_excel("eeg_data_expanded.xlsx")
eeg_data["freqs"] = eeg_data["freqs"].apply(safe_parse)
eeg_data["power"] = eeg_data["power"].apply(safe_parse)

# Remove rows with invalid or empty frequency/power data
# eeg_data = eeg_data[eeg_data["freqs"].notnull() & eeg_data["power"].notnull()]
if eeg_data.empty:
    raise ValueError("The cleaned EEG data is empty. Check the input file for issues.")

# **Select Data for a Specific Subject**
subject_id = "S164W1"
subject_data = eeg_data[eeg_data["id"] == subject_id]
if subject_data.empty:
    raise ValueError(f"No data found for Subject ID: {subject_id}")

# **Part 1: Plot Power vs. Frequency**
def plot_power_vs_frequency(subject_data, region="F7"):
    """Plot Power vs Frequency for a specific EEG region."""
    region_data = subject_data[subject_data["region"] == region]
    if region_data.empty:
        raise ValueError(f"No data found for region {region}")
    
    # Get frequency and power data
    freqs = region_data["freqs"].iloc[0]
    power = region_data["power"].iloc[0]
    
    # Plot Power vs Frequency
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, power, label=f"Region: {region}")
    plt.title(f"Power vs Frequency - Region {region}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.show()

# Call the function for Power vs Frequency
plot_power_vs_frequency(subject_data, region="O2")

print(subject_data[subject_data["region"] == "O2"])


# **Part 2: Plot Microvolts vs Time**
def plot_microvolts_vs_time(subject_data, sampling_rate=128):
    """Plot EEG microvolts vs time for all channels."""
    # Prepare the frequency data for reshaping
    freqs_list = subject_data["freqs"].to_list()
    if not all(isinstance(freqs, list) for freqs in freqs_list):
        raise ValueError("One or more 'freqs' entries are invalid or not a list.")
    
    # Reshape data
    reshaped_data = pd.DataFrame(
        np.array(freqs_list).T,  # Transpose to align time points
        columns=subject_data["region"].values
    )
    
    # Add time column
    time_points = reshaped_data.shape[0]
    reshaped_data["Time (s)"] = np.arange(0, time_points / sampling_rate, 1 / sampling_rate)
    
    # Plot a single channel
    channel = "F7"  # Replace with the desired channel
    plt.figure(figsize=(8, 5))
    plt.plot(reshaped_data["Time (s)"], reshaped_data[channel])
    plt.title(f"EEG Data - Channel {channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.grid()
    plt.show()
    
    # Plot all channels
    plt.figure(figsize=(10, 20))
    for i, channel in enumerate(subject_data["region"].values):
        plt.subplot(len(subject_data), 1, i + 1)
        plt.plot(reshaped_data["Time (s)"], reshaped_data[channel])
        plt.title(f"Channel {channel}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (μV)")
        plt.tight_layout()
    plt.show()

# Call the function for Microvolts vs Time
plot_microvolts_vs_time(subject_data)