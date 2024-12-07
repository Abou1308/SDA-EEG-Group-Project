import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from data_preprocessing import preprocess

# Step 1: Preprocess the data
df1 = preprocess()

# Step 2: Group by patient ID
patient_groups = df1.groupby('id')


for i in range(5):  # Check the first 5 rows
    print(f"Row {i} length:", len(df1.loc[i, 'eeg_data']))

# Step 3: Loop through each patient
for patient_id, group in patient_groups:
    print(f"Processing patient: {patient_id}")
    
    # Combine data for all 16 channels
    eeg_data_list = group['eeg_data'].tolist()  # Extract EEG data for all rows (16 rows)
    eeg_data = np.array(eeg_data_list)  # Combine into a 2D array
    # eeg_data = eeg_data / 20 #normalise

    print("EEG data range:", eeg_data.min(), eeg_data.max())
    # eeg_data = eeg_data.T  # Transpose to (n_channels, n_samples)
    
    print("EEG data shape:", eeg_data.shape)  # Should be (16, n_samples)
    print("EEG data range:", eeg_data.min(), eeg_data.max())  # Check amplitude range

    # Verify data shape
    print(f"EEG data shape for patient {patient_id}: {eeg_data.shape}")
    
    # Step 4: Create MNE Info object
    sfreq = 128  # Replace with your actual sampling frequency
    ch_names = [f"Ch{i+1}" for i in range(16)]  # Create channel names for 16 channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * 16)
    
    # Step 5: Create the RawArray
    raw = mne.io.RawArray(eeg_data, info)
    print(raw)
    
    # Step 6: Visualize the EEG data
    raw.plot_psd(fmin=1, fmax=64, spatial_colors=True, average=True)
    
    #If we ever have stimulus we can detect spikes
    # events = mne.find_events(raw)  # Detect events
    # raw.plot(events=events, event_color='red')
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
    # raw.plot(duration=10, n_channels=16)  # Plot all 16 channels for a 60-second duration
    plt.show()