import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import json #for robustness in deserializing
from data_preprocessing import preprocess

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
eeg_data = preprocess()
eeg_data["freqs"] = eeg_data["freqs"].apply(safe_parse)
eeg_data["power"] = eeg_data["power"].apply(safe_parse)

# **Select Data for a Specific Subject**
subject_id = "S164W1"
subject_data = eeg_data[eeg_data["id"] == subject_id]

# **Part 1: Plot Power vs. Frequency**
def plot_power_vs_frequency(subject_data, region="F8"):
    """Plot Power vs Frequency for a specific EEG region."""
    region_data = subject_data[subject_data["region"] == region]
    if region_data.empty:
        raise ValueError(f"No data found for region {region}")
    
    # Get frequency and power data
    freqs = region_data["freqs"].iloc[0]
    power = region_data["power"].iloc[0]
    
    # Filter out negative frequencies
    filtered_data = [(f, p) for f, p in zip(freqs, power) if f >= 0]
    filtered_freqs, filtered_power = zip(*filtered_data)

    # Plot Power vs Frequency
    plt.figure(figsize=(8, 5))
    plt.plot(filtered_freqs, filtered_power, label=f"Region: {region}")
    plt.title(f"Power vs Frequency - Region {region}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.show()

# Call the function for Power vs Frequency
for region in ["F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]:
    plot_power_vs_frequency(subject_data, region)