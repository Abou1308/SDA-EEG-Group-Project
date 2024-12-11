import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from data_preprocessing import preprocess

# Preprocess data (assuming it returns a DataFrame)
df = preprocess()

# Extract power data for each patient and region
def extract_power(df, patient_id, frequency_band):
    # Filter patient data
    patients = df[df['id'] == patient_id]
    power_data = []


    # Loop through regions and get the data
    for region in patients['region'].unique():
        region_data = patients[patients['region'] == region]
        power_data.append(region_data[frequency_band].values[0])

    power_df = pd.DataFrame(power_data).transpose()

    # Handle NaN values
    if power_df.isnull().values.any():
        print(f"Warning: Missing values detected in power data for patient {patient_id}")
        # Replace empty values with 0
        power_df = power_df.fillna(0)

    return power_df

# Calculate coherence between regions (pairwise)
def calculate_coherence_regions(power_df, sampling_freq):
    # Initialize empty matrix for coherence values
    coherence_matrix_values = np.zeros((len(power_df.columns), len(power_df.columns)))

    # Calculate coherence for each pair of regions
    for i, region_1 in enumerate(power_df.columns):
        for j, region_2 in enumerate(power_df.columns):
            # Compute coherence between two regions (channels)
            _, coherence_values = signal.coherence(power_df[region_1], power_df[region_2], fs=128, nperseg=256)
            coherence_matrix_values[i, j] = np.mean(coherence_values)  # Take the mean of coherence values
    return coherence_matrix_values

# Average coherence across multiple patients
def average_coherence(patient_ids, wave_option='power_beta'):
    # List comprehension to calculate coherence matrices for all patients
    coherence_matrices = [
        calculate_coherence_regions(extract_power(df, patient_id, wave_option), sampling_freq=128)
        for patient_id in patient_ids
    ]
    # Calculate the average coherence matrix by stacking all matrices and taking the mean
    average_coherence_matrix = np.mean(coherence_matrices, axis=0)

    # Return as a DataFrame
    return pd.DataFrame(average_coherence_matrix)


# Plot coherence matrices for healthy vs schizophrenic patients
def plot_correlation_matrix(healthy_patients, schizo_patients, wave_option='power_beta'):
    # Calculate average coherence for healthy and schizophrenic patients
    avg_coherence_healthy = average_coherence(healthy_patients, wave_option=wave_option)
    avg_coherence_schizo = average_coherence(schizo_patients, wave_option=wave_option)

    # Calculate coherence difference (healthy - schizo)
    coherence_difference = avg_coherence_healthy - avg_coherence_schizo

    regions = avg_coherence_healthy.columns.tolist()  # Get region names from columns

    # Plot for healthy patients
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_coherence_healthy, annot=True, cmap='Oranges', fmt='.2f', xticklabels=regions, yticklabels=regions)
    plt.title(f"Average Coherence Matrix for Healthy Patients ({wave_option.capitalize()})")
    plt.xlabel('Regions (Channels)')
    plt.ylabel('Regions (Channels)')
    plt.show()

    # Plot for schizophrenic patients
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_coherence_schizo, annot=True, cmap='Oranges', fmt='.2f', xticklabels=regions, yticklabels=regions)
    plt.title(f"Average Coherence Matrix for Schizophrenic Patients ({wave_option.capitalize()})")
    plt.xlabel('Regions (Channels)')
    plt.ylabel('Regions (Channels)')
    plt.show()

    # Plot coherence difference (healthy - schizo)
    plt.figure(figsize=(10, 8))
    sns.heatmap(coherence_difference, annot=True, cmap='Oranges', fmt='.2f', xticklabels=regions, yticklabels=regions)
    plt.title('Coherence Difference (Healthy - Schizophrenic)')
    plt.xlabel('Regions (Channels)')
    plt.ylabel('Regions (Channels)')
    plt.show()

# Example: Plot the correlation matrices
healthy_patients = df[df['schizo'] == 0]['id'].unique().tolist()
schizo_patients = df[df['schizo'] == 1]['id'].unique().tolist()
plot_correlation_matrix(healthy_patients, schizo_patients, wave_option='power_beta')
