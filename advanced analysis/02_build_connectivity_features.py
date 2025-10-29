import os
import sys
import scipy.io
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs
import warnings
from tqdm import tqdm # Progress bar
from pandas.errors import SettingWithCopyWarning

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
try:
    from data_loader import load_master_logbook
except ImportError:
    print("Error: Could not import load_master_logbook.")
    sys.exit(1)

MAT_FILE_DIR = os.path.join(project_dir, 'eeg_mat_files')
OUTPUT_FILE = os.path.join(project_dir, 'connectivity_features.csv') # Save in main dir
SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'
BANDS = {
    'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13],
    'beta': [13, 30], 'gamma': [30, 45]
}
CONN_METHOD = 'coh' # Coherence
EPOCH_DURATION = 5.0 # seconds
NUM_CHANNELS = 6

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", message="fmin=* Hz corresponds to * < 5 cycles", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning) # General user warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


def calculate_connectivity(eeg_segment, srate=SAMPLING_RATE):
    """Calculates connectivity matrices for a single EEG segment."""
    num_channels = eeg_segment.shape[0]
    if num_channels != NUM_CHANNELS: return None # Basic check

    ch_names = [f'Ch{i+1}' for i in range(num_channels)]
    ch_types = ['eeg'] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False) # uV to V
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)

    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, preload=True, verbose=False)
        if len(epochs) == 0: return None
    except ValueError:
         return None

    conn_matrices = {}
    for band_name, (fmin, fmax) in BANDS.items():
        try:
            con = spectral_connectivity_epochs(
                epochs, method=CONN_METHOD, mode='multitaper',
                sfreq=srate, fmin=fmin, fmax=fmax,
                faverage=True, n_jobs=1, verbose=False
            )
            conn_data = con.get_data(output='dense').squeeze()
            if conn_data.shape == (num_channels, num_channels):
                 conn_matrices[band_name] = conn_data
            else:
                 conn_matrices[band_name] = np.full((num_channels, num_channels), np.nan)
        except Exception as e:
            conn_matrices[band_name] = np.full((num_channels, num_channels), np.nan)

    return conn_matrices

def extract_lower_triangle(matrix):
    """Extracts unique connection values from a symmetric matrix."""
    # indices of the lower triangle (excluding diagonal k=0)
    rows, cols = np.tril_indices_from(matrix, k=-1)
    return matrix[rows, cols] # Returns a flat array of 15 values

def create_connectivity_dataset():
    """Loops through dreams, calculates connectivity, and saves features."""
    logbook = load_master_logbook()
    if logbook is None: return

    # Use the bug-fixed 3-class mapping for labels
    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion: return 'Negative'
        if 'happy' in emotion: return 'Positive'
        if 'calm' in emotion: return 'Calm'
        return None
    logbook['Emotion_Label'] = logbook['Dream_emotion'].apply(map_emotion)

    all_features_list = []
    print(f"\nStarting Connectivity Feature Extraction ({CONN_METHOD})")

    for index, row in tqdm(logbook.iterrows(), total=logbook.shape[0]):
        participant_id = row['Online_id']
        emotion_label = row['Emotion_Label']
        video_type = row['Video_type'] # Keep video type as well

        # Find and load .mat file
        mat_file_name = None
        try:
            for file in os.listdir(MAT_FILE_DIR):
                if participant_id in file and file.endswith('.mat'):
                    mat_file_name = file
                    break
        except FileNotFoundError: continue # Skip if dir not found
        if mat_file_name is None: continue # Skip if file not found

        mat_file_path = os.path.join(MAT_FILE_DIR, mat_file_name)
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            eeg_segment = mat_data[EEG_VARIABLE_NAME]
            if eeg_segment.shape[1] < SAMPLING_RATE * EPOCH_DURATION: continue # Skip short segments
        except Exception: continue # Skip load errors

        # Calculate connectivity
        conn_matrices = calculate_connectivity(eeg_segment)
        if conn_matrices is None: continue # Skip calculation errors

        # Flatten matrices and store features
        dream_features = {'Online_id': participant_id, 'Emotion_Label': emotion_label, 'Video_type': video_type}
        valid_calculation = True
        for band, matrix in conn_matrices.items():
            if np.isnan(matrix).all() or matrix.shape != (NUM_CHANNELS, NUM_CHANNELS):
                 valid_calculation = False
                 break # Skip dream if any band failed
            flat_conn = extract_lower_triangle(matrix) # Get 15 values
            # Create feature names like 'delta_Ch1_Ch2', 'delta_Ch1_Ch3', ...
            rows, cols = np.tril_indices(NUM_CHANNELS, k=-1)
            for i, val in enumerate(flat_conn):
                feature_name = f"{band}_Ch{rows[i]+1}_Ch{cols[i]+1}"
                dream_features[feature_name] = val

        if valid_calculation:
            all_features_list.append(dream_features)

    print(f"\nSuccessfully extracted connectivity features for {len(all_features_list)} dreams.")

    # Save to CSV
    if all_features_list:
        features_df = pd.DataFrame(all_features_list)
        try:
            features_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSuccessfully saved connectivity features to '{OUTPUT_FILE}'!")
            print("\nPreview:")
            print(features_df.head())
        except Exception as e:
            print(f"Error saving features to CSV: {e}")
    else:
        print("\nNo connectivity features were successfully extracted.")


if __name__ == '__main__':
    create_connectivity_dataset()