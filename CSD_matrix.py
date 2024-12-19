import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import csd

import seaborn as sns


def cross_spectral_matrix(patient_id, dataframe, wave_option='power_beta', surrogate=False):
    """Function mainly used for returns the CSD Matrices"""
    patient_data = dataframe[dataframe['id'] == patient_id]

    #Create an empty list to hold the power data for each region (channel)
    power_data = []

    sampling_freq = 128

    #Loops through the regions (channels) and extract power data
    for region in patient_data['region'].unique():

        #Filters power data for the current region
        region_data = patient_data[patient_data['region'] == region]
        
        #Append the power data (list) for this region to the list
        power_data.append(region_data[wave_option].values[0])  # Assuming 'power' is a list

    #Converts the list of power data (regions x time points) into a DataFrame
    power_df = pd.DataFrame(power_data).transpose()

    #Checks for missing values
    if power_df.isnull().values.any():
        print(f"Warning: Missing values detected in power data for patient {patient_id}")
        power_df = power_df.fillna(0)  # Fill NaN with 0

    channels = power_df.columns
    csd_matrix = np.zeros((len(channels), len(channels)), dtype=complex)

    #Computes CSD Matrix
    for i, ch1 in enumerate(channels):
        for j, ch2 in enumerate(channels):

            #Computes CSD between two channels
            freqs, csd_values = csd(power_df[ch1], power_df[ch2], fs=sampling_freq, nperseg=256)
            
            # Take the average value of the CSD (optional: could store full spectrum instead)
            csd_matrix[i, j] = np.mean(csd_values)

    #Normalizes CSD Matrix
    for i in range(len(channels)):
        for j in range(len(channels)):
            if i != j:
                csd_matrix[i, j] = csd_matrix[i, j] / np.sqrt(csd_matrix[i, i] * csd_matrix[j, j])
    
    np.fill_diagonal(csd_matrix, 1) #Explicitly sets the diagonals to 1
    
    #Returns the CSD matrix
    return csd_matrix, freqs
    

def average_csd(patient_ids, dataframe, wave_option='power_beta', surrogate=False):
    """Function that computes the average of CSD matrices across groups"""
    csd_sums = None
    count = 0

    #Loops through each patient and calculates their CSD matrix
    for patient_id in patient_ids:
        csd_matrix, _ = cross_spectral_matrix(patient_id, dataframe, wave_option, surrogate=surrogate)
        
        #Add the current CSD matrix to the cumulative sum
        if csd_sums is None:
            csd_sums = csd_matrix
        else:
            csd_sums += csd_matrix
        count += 1

    #Retrieves the magnitudes only for comparison
    average_csd_matrix = np.abs(csd_sums / count)

    region_labels = ["F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    
    average_csd_df = pd.DataFrame(average_csd_matrix, columns=region_labels, index=region_labels)

    return average_csd_df