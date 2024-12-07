import pandas as pd
import numpy as np
import ast
# Read CSV file
file_path = "eeg_data_processed.csv"
df = pd.read_csv(file_path)

# Convert string-encoded lists back to numerical arrays
list_columns = ['freqs', 'power', "freqs_theta", "freqs_alpha","freqs_beta",]
for col in list_columns:
    df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)))

# Function to calculate the Coherence
def calculate_coherence(fft1, fft2):
    # Calculate the power
    Pxx = np.abs(fft1) ** 2
    Pyy = np.abs(fft2) ** 2

    # Calculate Cross-spectral Density (CSD)
    csd = fft1 * np.conj(fft2)

    # Calculate Coherence
    coherence = np.abs(csd) ** 2 / (Pxx * Pyy)

    return coherence

# Calculate coherence for all regions for all patients
def calculate_coherence_regions(region_1, region_2):
    coherence_results = []

    # Filter to only keep wanted regions
    reg1 = df[df['region'] == region_1]
    reg2 = df[df['region'] == region_2]

    common_id = set(reg1['id']).intersection(reg2['id'])

    # Calculate coherence for all patients
    for patient in common_id:
        # Extract FFT data for the patient
        fft_1_alpha = np.array(reg1[reg1['id'] == patient]['freqs_alpha'].values[0])
        fft_2_alpha = np.array(reg2[reg2['id'] == patient]['freqs_alpha'].values[0])

        fft_1_beta = np.array(reg1[reg1['id'] == patient]['freqs_beta'].values[0])
        fft_2_beta = np.array(reg2[reg2['id'] == patient]['freqs_beta'].values[0])

        fft_1_theta = np.array(reg1[reg1['id'] == patient]['freqs_theta'].values[0])
        fft_2_theta = np.array(reg2[reg2['id'] == patient]['freqs_theta'].values[0])

        # Calculate coherence for all frequencies
        coherence_alpha = calculate_coherence(fft_1_alpha, fft_2_alpha)
        coherence_beta = calculate_coherence(fft_1_beta, fft_2_beta)
        coherence_theta = calculate_coherence(fft_1_theta, fft_2_theta)

        freqs = np.array(reg1[reg1['id'] == patient]['freqs'].values[0])

        coherence_results.append({
            "id": patient,
            "region_1": region_1,
            "region_2": region_2,
            "frequencies": freqs.tolist(),
            "coherence_alpha": coherence_alpha.tolist(),
            "coherence_beta": coherence_beta.tolist(),
            "coherence_theta": coherence_theta.tolist()
        })

    return pd.DataFrame(coherence_results)

regions = [
    "F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4",
    "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

# Calculate coherence for all the regions
all_coherence = []
# Loop over region 1 and 2
for i, region_1 in enumerate(regions):
    for region_2 in regions[i+1:]:
        # Calcualte coherence
        region_coherence = calculate_coherence_regions(region_1, region_2)
        all_coherence.append(region_coherence)
final_coherence_df = pd.concat(all_coherence, ignore_index = True)

final_coherence_df.to_csv("all_region_coherence.csv", index=False)
