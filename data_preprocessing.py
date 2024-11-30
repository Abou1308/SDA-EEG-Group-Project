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
    with open(file_path, "r") as file:
        lines = file.readlines()
    rows = []
    file_id = os.path.basename(file_path).replace(".eea", "")
    for i, region in enumerate(regions):
        start = i * 7680
        end = start + 7680
        eeg_data = [float(value.strip()) for value in lines[start:end]]
        eeg_data_np = np.array(eeg_data)

        fft_result = np.fft.fft(eeg_data_np)
        freqs = np.fft.fftfreq(len(eeg_data_np), 1/128)

        fft_magnitude = np.abs(fft_result)

        # When adding data here, we add a new row per region
        rows.append({
            "id": file_id,
            "schizo": schizo_label,
            "region": region,
            "eeg_data": eeg_data,
            "freqs": freqs,  # Store the frequency bins
            "power": fft_magnitude  # Store the magnitude (power) of the FFT
        })

    return rows


def process_directory(directory, schizo_label):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".eea"):
            file_path = os.path.join(directory, file_name)
            data.extend(process_file(file_path, schizo_label))
    return data


def check_file_exists(filename):
    # Get the absolute path of the current script's directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the file
    file_path = os.path.join(current_directory, filename)
    # Check if the file exists
    return os.path.isfile(file_path)

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
        patient_values.extend(row['power'])
        
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

if __name__ == "__main__":
    norm_data = process_directory(norm_dir, schizo_label=0)
    schizo_data = process_directory(schizo_dir, schizo_label=1)
    combined_data = norm_data + schizo_data
    df = pd.DataFrame(combined_data)
    if check_file_exists('eeg_data_processed.xlsx') == False:
        df.to_excel("eeg_data_processed.xlsx", index=False)

    # Normalization
    normalized_data = normalize_data(df)
    normalized_electrodes = normalize_electrodes(normalized_data)
    normalized_patients = normalize_patients(normalized_electrodes)
    if check_file_exists('eeg_data_normalized.xlsx') == False:
        normalized_patients.to_excel("eeg_data_normalized.xlsx", index=False)