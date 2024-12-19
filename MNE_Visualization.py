import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from data_preprocessing import preprocess

#Preprocess the data: Calculates Freqs & Power Ranges
df1 = preprocess()

#Group by patient ID
patient_groups = df1.groupby('id')

processed_healthy = False
processed_schizo = False


for patient_id, group in patient_groups:
    
    is_schizo = group['schizo'].iloc[0]

    if processed_healthy and processed_schizo:
        break

    if not processed_healthy and is_schizo == 0:
        print(f"Processing healthy patient: ID {patient_id}")
        print(f"Condition {is_schizo}")
        processed_healthy = True
    
    elif not processed_schizo and is_schizo == 1:
        print(f"Processing schizo patient: ID {patient_id}")
        print(f"Condition {is_schizo}")
        processed_schizo = True
    
    else:
        continue

    #Extract EEG data for all rows (16 rows)
    eeg_data_list = group['eeg_data'].tolist()
    eeg_data = np.array(eeg_data_list)

    print("EEG data range:", eeg_data.min(), eeg_data.max()) 
    print("EEG data shape:", eeg_data.shape)  # Should be (16, n_samples)
    
    #Creates MNE Info object
    sfreq = 128 
    ch_names = [f"Ch{i+1}" for i in range(16)] 
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * 16)
    
    #Create the RawArray. Necessary for MNE
    raw = mne.io.RawArray(eeg_data, info)
    print(raw)

    #Applies band-pass filter between 4 and 30 Hz
    raw.filter(l_freq=4, h_freq=30)
    
    #Creates a Power Spectral Density Plot
    raw.plot_psd(fmin=4, fmax=30, spatial_colors=True, average=True)
    
    mapping = {
    "Ch1": "F7", "Ch2": "F3", "Ch3": "F4", "Ch4": "F8",
    "Ch5": "T3", "Ch6": "C3", "Ch7": "Cz", "Ch8": "C4",
    "Ch9": "T4", "Ch10": "T5", "Ch11": "P3", "Ch12": "Pz",
    "Ch13": "P4", "Ch14": "T6", "Ch15": "O1", "Ch16": "O2"}

    raw.rename_channels(mapping)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.plot_sensors(show_names=True, kind="topomap")
    raw.plot(duration=10, n_channels=16, scalings="auto", color={"eeg": "blue"})

    plt.show()