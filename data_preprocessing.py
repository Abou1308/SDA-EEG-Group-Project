# This file is not supposed to be ran on itself. The intended way to use this is to import the preprocess()
# function contained in this file, which would turn the .eea data into a usable pandas dataframe.

import os
import pandas as pd
import numpy as np

# Paths and mappings
norm_dir = "./norm_data"
schizo_dir = "./schizo_data"
regions = [
    "F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4",
    "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

def process_file(file_path, schizo_label):
    """
    Function processes an .eea file and returns the preprocessed eeg data after performing a fast fourier
    transform. It further divides the frequencies and powers based on the corresponding brain wave.
    """
    # Read lines
    with open(file_path, "r") as file:
        lines = file.readlines()
    rows = []
    file_id = os.path.basename(file_path).replace(".eea", "")

    # Each file has 16 regions. Each 7680 line combination is a different region, we loop through the file in 
    # intervals of 7680 to cover each region.
    for i, region in enumerate(regions):
        start = i * 7680
        end = start + 7680
        eeg_data = [float(value.strip()) for value in lines[start:end]]
        eeg_data_np = np.array(eeg_data)

        # Perform fast fourier transform and perform a simple list comprehension for formatting
        fft_result = np.fft.fft(eeg_data_np)
        freqs = np.fft.fftfreq(len(eeg_data_np), 1/128)
        float_freqs = [float(x) for x in freqs]
        fft_magnitude = np.abs(fft_result)
        float_power = [float(x) for x in fft_magnitude]
        # When adding data here, we add a new row per region
        rows.append({
            "id": file_id,
            "schizo": schizo_label,
            "region": region,
            "eeg_data": eeg_data,
            "freqs":float_freqs,
            "power":float_power,
            "freqs_theta": float_freqs[240:481],
            "power_theta": float_power[240:481],
            "freqs_alpha": float_freqs[480:721],
            "power_alpha": float_power[480:721],
            "freqs_beta":float_freqs[720:1801],
            "power_beta":float_power[720:1801]
        })

    return rows

def process_directory(directory, schizo_label):
    """
    Processes each file in a folder, which in our case is norm_data for all healthy patients and
    schizo_data for all schizophrenic patients. schizo_label should contain whether the folder is for 
    schizophrenic or healthy patients.
    """
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".eea"):
            file_path = os.path.join(directory, file_name)
            data.extend(process_file(file_path, schizo_label))
    return data

def normalize_data(input_data):
    # Calculate mean and stdev of each electrode
    normalized_data = []

    for index, row in input_data.iterrows():
        power = row['power']
        power = np.array(power)

        mean_power = np.mean(power)
        std_power = np.std(power)

        # Apply z score normalization
        normalized_power = (power - mean_power) / std_power

        normalized_data.append({
            "id": row['id'],
            "schizo": row['schizo'],
            "region": row['region'],
            "freqs": row['freqs'],
            "normalized_power": normalized_power.tolist()
        })

    normalized_df = pd.DataFrame(normalized_data)
    return normalized_df

# Normalize across electrodes for each patient
def normalize_electrodes(normal_data):
    normalized_data = []

    for index, row in normal_data.iterrows():
        normal_power = row['normalized_power']
        normal_power = np.array(normal_power)

        mean_normal_power = np.mean(normal_power)
        std_normal_power = np.std(normal_power)

        normal_electrode = (normal_power - mean_normal_power) / std_normal_power
        normalized_data.append({
            "id": row['id'],
            "schizo": row['schizo'],
            "region": row['region'],
            "freqs": row['freqs'],
            "normalized_power": normal_electrode.tolist()
        })

    normalized_electrodes_df = pd.DataFrame(normalized_data)
    return normalized_electrodes_df

# Normalize across all patients
def normalize_patients(normal_data):
    # Get the data from all patients
    patient_values = []
    for index,row in normal_data.iterrows():
        patient_values.extend(row['normalized_power'])

    patient_values = np.array(patient_values)
    total_mean = np.mean(patient_values)
    total_std = np.std(patient_values)

    # Normalize data between each patients
    normalized_data = []
    for index, row in normal_data.iterrows():
        power = np.array(row['normalized_power'])
        normal_power = (power - total_mean) / total_std

        normalized_data.append({
            "id": row['id'],
            "schizo": row['schizo'],
            "region": row['region'],
            "freqs": row['freqs'],
            "normalized_power": normal_power.tolist()
        })

    normalized_df = pd.DataFrame(normalized_data)
    return normalized_df

def preprocess(normalization=False):
    """
    The main function which when called, should return a pandas dataframe with all required information.
    """
    # Define directory and process them both
    norm_dir = "./norm_data"
    schizo_dir = "./schizo_data"
    norm_data = process_directory(norm_dir, schizo_label=0)
    schizo_data = process_directory(schizo_dir, schizo_label=1)
    # Combine them en return it as a pandas dataframe.
    combined_data = norm_data + schizo_data
    df = pd.DataFrame(combined_data)

    if normalization:
        normalized_data = normalize_data(df)
        normalized_electrodes = normalize_electrodes(normalized_data)
        normalized_patients = normalize_patients(normalized_electrodes)
        return normalized_patients
    else:
        return df

