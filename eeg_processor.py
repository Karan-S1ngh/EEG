import os
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import mne
from data_loader import load_master_logbook

MAT_FILE_DIR = 'eeg_mat_files'
SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'

# Define the EEG frequency bands
BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 45]
}

def _calculate_psd_features(eeg_segment, srate=SAMPLING_RATE):
    """
    Cleans an EEG segment and calculates Power Spectral Density (PSD) features
    FOR EACH CHANNEL SEPARATELY.
    """
    num_channels = eeg_segment.shape[0]
    ch_names = [f'EEG {i+1}' for i in range(num_channels)]
    ch_types = ['eeg'] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
    
    raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False)
    raw.set_montage('standard_1020', on_missing='ignore')
    
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
    
    spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45.0, n_fft=srate, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    
    features = {}
    # Loop through each channel
    for ch_index in range(num_channels):
        # Get the channel name (e.g., 'EEG 1')
        ch_name = f'ch{ch_index + 1}' 
        
        # Calculate power in each band for *this channel only*
        for band_name, (fmin, fmax) in BANDS.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # Get PSD for this channel, find power in this band
            channel_psd = psd[ch_index, :]
            band_power = np.log10(np.mean(channel_psd[idx_band]))
            
            # Create a unique feature name, e.g., 'ch1_alpha_power'
            features[f'{ch_name}_{band_name}_power'] = band_power
    
    return features

def extract_eeg_features(dream_row):
    """
    Main function to extract EEG features for a single dream (a row from the master_logbook).
    """
    participant_id = dream_row['Online_id']
    
    mat_file_name = None
    try:
        for file in os.listdir(MAT_FILE_DIR):
            if participant_id in file and file.endswith('.mat'):
                mat_file_name = file
                break
    except FileNotFoundError:
        print(f"Error: Directory not found: '{MAT_FILE_DIR}'.")
        return None

    if mat_file_name is None:
        print(f"Warning: No .mat file found containing ID {participant_id}. Skipping.")
        return None
    
    mat_file_path = os.path.join(MAT_FILE_DIR, mat_file_name)
        
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        eeg_segment = mat_data[EEG_VARIABLE_NAME]
        
        if eeg_segment.shape[1] == 0:
            print(f"Warning: EEG segment is empty for {participant_id}. Skipping.")
            return None
            
    except KeyError:
        print(f"The variable '{EEG_VARIABLE_NAME}' was not found in {mat_file_name}.")
        print(f"The variables I *did* find are: {list(mat_data.keys())}")
        return None
    except Exception as e:
        print(f"Error loading {mat_file_name}: {e}")
        return None
        
    features = _calculate_psd_features(eeg_segment)
    
    features['Online_id'] = participant_id
    features['Dream_emotion'] = dream_row['Dream_emotion']
    features['Video_type'] = dream_row['Video_type']
    
    return features


if __name__ == '__main__':  
    print("Testing eeg_processor.py (V6 - 30 Features)")
    
    logbook = load_master_logbook()
    
    if logbook is not None:
        test_row = logbook.dropna(
            subset=['Start_recording', 'REM_period_start_time', 'End_time_of_REM_period']
        ).iloc[0]
        
        print(f"\nAttempting to process first valid dream from: {test_row['Online_id']}")
        
        if not os.path.exists(MAT_FILE_DIR):
            print(f"\nERROR: Folder '{MAT_FILE_DIR}' not found.")
        else:
            features = extract_eeg_features(test_row)
            
            if features:
                print("\nSUCCESSFULLY EXTRACTED FEATURES!")
                print(f"Found {len(features) - 3} features (e.g., ch1_delta_power).")
                print(features)
            else:
                print("\nCould not extract features for test row.")