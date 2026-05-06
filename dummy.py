import pandas as pd
import numpy as np
import re
import os
import glob

# ==========================================
# 1. Data Processing
# ==========================================

def read_csi_waveform(file_path):
    '''
    Read CSI waveform data from a CSV file and return it as a complex NumPy array.
    '''
    df = pd.read_csv(file_path)

    sub_indices = set()
    for col in df.columns:
        match = re.search(r'Sub_(\d+)_Real', col)
        if match:
            sub_indices.add(int(match.group(1)))
    
    sorted_indices = sorted(list(sub_indices))
    
    if not sorted_indices:
        raise ValueError(f"No valid CSI subcarrier columns found in {file_path}")

    num_packets = len(df)
    num_subcarriers = len(sorted_indices)
    
    csi_matrix = np.zeros((num_packets, num_subcarriers), dtype=complex)
    
    for matrix_col_idx, sub_idx in enumerate(sorted_indices):
        real_col = f'Sub_{sub_idx}_Real'
        imag_col = f'Sub_{sub_idx}_Imag'
        csi_matrix[:, matrix_col_idx] = df[real_col].values + 1j * df[imag_col].values
        
    return csi_matrix

def get_dataset(root_path, device_ids):
    '''
    Loads CSI data and returns features (Amplitude, Phase).
    Returns X shape: (N, Subcarriers, 2)
    '''
    X_list = []
    y_list = []

    print(f"Scanning dataset at: {root_path}")

    for label_idx, dev_id in enumerate(device_ids):
        device_folder = os.path.join(root_path, dev_id)
        search_pattern = os.path.join(device_folder, "*.csv")
        files = glob.glob(search_pattern)

        if not files:
            print(f"Warning: No CSV files found in {device_folder}")
            continue

        print(f"Found {len(files)} files for device '{dev_id}'")

        for file_path in files:
            try:
                csi_data = read_csi_waveform(file_path)
                if csi_data.shape[0] == 0:
                    continue

                X_list.append(csi_data)
                
                # Create labels
                labels = np.full(csi_data.shape[0], label_idx, dtype=int)
                y_list.append(labels)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not X_list:
        raise ValueError("No data loaded. Check your root_path and folder structure.")

    # Concatenate all complex data first
    X_complex = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Extract Amplitude and Unwrapped Phase
    amplitude = np.abs(X_complex)
    raw_phase = np.angle(X_complex)
    unwrapped_phase = np.unwrap(raw_phase, axis=-1)

    # Stack to (N, Subcarriers, 2)
    # NOTE: Changed to axis=2 so shape is (N, 52, 2). 
    # This matches RFNet's first layer permute(0,2,1) -> (N, 2, 52).
    X = np.stack((amplitude, unwrapped_phase), axis=2)

    print("-" * 30)
    print(f"Dataset loaded successfully.")
    print(f"Total packets: {X.shape[0]}")
    print(f"Total subcarriers: {X.shape[1]}")
    
    return X, y


root_path = './csi'
devices = ['device_1', 'device_2'] 

X, y = get_dataset(root_path, devices)