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

# Process a signle file
def process_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Enumerate over the regions
    region_data = {}
    for i, region in enumerate(regions):
        start = i * 7680
        end = start + 7680
        region_data[region] = [float(value.strip()) for value in lines[start:end]]
    return region_data

# Now process all files in the directory with this function
def process_directory(directory, schizo_label):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".eea"):
            file_path = os.path.join(directory, file_name)
            region_data = process_file(file_path)
            region_data["id"] = file_name.replace(".eea", "")
            region_data["schizo"] = schizo_label # Indicataes whether the data is from someone schizo
            data.append(region_data)
    return data


norm_data = process_directory(norm_dir, schizo_label=0)
schizo_data = process_directory(schizo_dir, schizo_label=1)
combined_data = norm_data + schizo_data
df = pd.DataFrame(combined_data)


df = df[["id", "schizo"] + regions]
print(df.head())
df.to_csv("eeg_data.csv", index=False)

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

        # apply fourrier transform to get the data in frequency
        fft_result = np.fft.fft(eeg_data_np)
        freqs = np.fft.fftfreq(len(eeg_data_np), 1/128)

        fft_magnitude = np.abs(fft_result)

        # When adding data here, we add a new row per region
        rows.append({
            "id": file_id,
            "schizo": schizo_label,
            "region": region,
            "eeg_data": eeg_data,
            "freqs": freqs.tolist(),  # Store the frequency bins
            "power": fft_magnitude.tolist()  # Store the magnitude (power) of the FFT
        })
    return rows

def process_directory(directory, schizo_label):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".eea"):
            file_path = os.path.join(directory, file_name)
            data.extend(process_file(file_path, schizo_label))
    return data


norm_data = process_directory(norm_dir, schizo_label=0)
schizo_data = process_directory(schizo_dir, schizo_label=1)
combined_data = norm_data + schizo_data
df = pd.DataFrame(combined_data)


print(df.head())
df.to_excel("eeg_data_expanded.xlsx", index=False)
