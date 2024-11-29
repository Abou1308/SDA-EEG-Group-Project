import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import ast

# Load the EEG data CSV file
# file_path = "eeg_data_expanded.xlsx"
# eeg_data = pd.read_csv(file_path)
eeg_data = pd.read_excel("eeg_data_expanded.xlsx")

# # Debug: Print column names and sample data
# print("Columns:", eeg_data.columns)
# print(eeg_data.head())

# # Check for expected columns
# required_columns = {"id", "schizo", "region", "eeg_data", "freqs", "power"}
# # if not required_columns.issubset(freqs.columns):
# #     raise ValueError(f"CSV file is missing one or more required columns: {required_columns}")

# # Ensure 'eeg_data' column is properly formatted as lists
# eeg_data["freqs"] = eeg_data["freqs"].apply(eval)  # Convert string representation to Python lists

# Function to safely parse 'freqs' column
def safe_parse(value):
    try:
        if isinstance(value, str):
            return ast.literal_eval(value)  # Parse valid string lists
        elif isinstance(value, (list, np.ndarray)):
            return value  # Keep lists as-is
        else:
            return None  # Return None for invalid entries
    except (ValueError, SyntaxError) as e:
        print(f"Malformed 'freqs' entry: {value} -> Error: {e}")
        return None

# Apply safe parsing to 'freqs'
eeg_data["freqs"] = eeg_data["freqs"].apply(safe_parse)
print(f"Rows after cleaning 'freqs': {len(eeg_data)}")
# print(eeg_data[eeg_data["freqs"].isnull()])  # Print rows with invalid 'freqs'
# Drop rows with invalid or malformed 'freqs'
eeg_data = eeg_data[eeg_data["freqs"].notnull()]

#added
if eeg_data.empty:
    raise ValueError("The cleaned EEG data is empty. Check the input file for issues.")

# Debug: Print clean DataFrame
print("Cleaned DataFrame:")
print(eeg_data.head())

# Select data for a specific subject
subject_id = "S164W1"  # Replace with the desired subject ID
subject_data = eeg_data[eeg_data["id"] == subject_id]

#added
if subject_data.empty:
    raise ValueError(f"No data found for Subject ID: {subject_id}")

# Debug: Check data for the selected subject
print(f"Data for Subject {subject_id}:")
print(subject_data)

# Ensure 'freqs' has consistent shapes
freqs_list = subject_data["freqs"].to_list()
if not all(isinstance(freqs, list) for freqs in freqs_list):
    raise ValueError("One or more 'freqs' entries are invalid or not a list.")

# Reshape the data into a single DataFrame with channels as columns
try:
    reshaped_data = pd.DataFrame(
        np.array(freqs_list).T,  # Transpose to align time points
        columns=subject_data["region"].values  # Use regions as column names
    )
except ValueError as e:
    raise ValueError(f"Error reshaping data: {e}")

# Add a time column (assuming 128 Hz sampling rate)
sampling_rate = 128
time_points = reshaped_data.shape[0]
reshaped_data["Time (s)"] = np.arange(0, time_points / sampling_rate, 1 / sampling_rate)

# Debug: Display the reshaped data
print("Reshaped Data:")
print(reshaped_data.head())

# Plot a single channel
channel = "F7"  # Replace with the desired channel
plt.plot(reshaped_data["Time (s)"], reshaped_data[channel])
plt.title(f"EEG Data - Channel {channel}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.show()

# Plot all channels
plt.figure(figsize=(10, 20))
for i, channel in enumerate(subject_data["region"].values):
    plt.subplot(len(subject_data), 1, i + 1)
    plt.plot(reshaped_data["Time (s)"], reshaped_data[channel])
    plt.title(f"Channel {channel}")
    plt.tight_layout()
plt.show()

# Define a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 1 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)

# Apply filter to a specific channel
filtered_data = bandpass_filter(reshaped_data["F7"], lowcut=0.5, highcut=30, fs=128)

# Plot the filtered data
plt.plot(reshaped_data["Time (s)"], filtered_data)
plt.title("Filtered EEG Data - Channel F7")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.show()

