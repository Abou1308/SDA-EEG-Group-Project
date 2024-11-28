# Normalize data within each electrode
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
    

normalized_data = normalize_data(df)
normalized_electrodes = normalize_electrodes(normalized_data)
normalized_patients = normalize_patients(normalized_electrodes)
normalized_patients.to_excel("eeg_data_normalized.xlsx", index=False)
