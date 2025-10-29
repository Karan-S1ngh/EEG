import os
import sys
import scipy.io
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
import warnings

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
try:
    from data_loader import load_master_logbook
except ImportError:
    print("Error: Could not import load_master_logbook. Make sure connectivity_processor.py is in a subfolder.")
    sys.exit(1)


MAT_FILE_DIR = os.path.join(project_dir, 'eeg_mat_files')
SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'
BANDS = {
    'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13],
    'beta': [13, 30], 'gamma': [30, 45]
}
CONN_METHOD = 'coh' # Coherence
EPOCH_DURATION = 5.0 # Increased epoch duration slightly

# Suppress RuntimeWarnings about epoch length for low frequencies
warnings.filterwarnings("ignore", message="fmin=* Hz corresponds to * < 5 cycles", category=RuntimeWarning)


def calculate_connectivity(dream_row):
    """
    Loads EEG, preprocesses, and calculates frequency-specific connectivity
    band by band. V3 fixes matrix shape issue.
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
        print(f"Warning: No .mat file found for {participant_id}. Skipping.")
        return None
    mat_file_path = os.path.join(MAT_FILE_DIR, mat_file_name)

    # Load EEG Data
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        eeg_segment = mat_data[EEG_VARIABLE_NAME]
        if eeg_segment.shape[1] < SAMPLING_RATE * EPOCH_DURATION:
             print(f"Warning: EEG segment too short for {participant_id}. Skipping.")
             return None
    except Exception as e:
        print(f"Error loading {mat_file_name}: {e}")
        return None

    # Preprocessing with MNE 
    num_channels = eeg_segment.shape[0]
    if num_channels != 6:
         print(f"Warning: Expected 6 channels, found {num_channels} for {participant_id}. Skipping.")
         return None
    ch_names = [f'Ch{i+1}' for i in range(num_channels)]
    ch_types = ['eeg'] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False) # uV to V
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)

    # Calculate Connectivity Band by Band 
    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, preload=True, verbose=False)
        if len(epochs) == 0:
            print(f"Warning: No valid epochs created for {participant_id}. Skipping.")
            return None
    except ValueError as e:
         print(f"Error creating epochs for {participant_id}: {e}. Skipping.")
         return None


    conn_matrices = {}
    print(f"\nCalculating {CONN_METHOD} connectivity for {participant_id}")

    for band_name, (fmin, fmax) in BANDS.items():
        print(f"Processing {band_name} band ({fmin}-{fmax} Hz)")
        try:
            con = spectral_connectivity_epochs(
                epochs, method=CONN_METHOD, mode='multitaper',
                sfreq=SAMPLING_RATE, fmin=fmin, fmax=fmax,
                faverage=True, # Average within the band
                n_jobs=1, verbose=False
            )
            # Remove the extra dimension using squeeze()
            conn_data = con.get_data(output='dense').squeeze()
            if conn_data.shape == (num_channels, num_channels):
                 conn_matrices[band_name] = conn_data
            else:
                 # Handle unexpected shape after squeeze
                 print(f"Warning: Unexpected matrix shape {conn_data.shape} for {band_name}. Filling with NaN.")
                 conn_matrices[band_name] = np.full((num_channels, num_channels), np.nan)


        except Exception as e:
            print(f"Error calculating {band_name} connectivity: {e}")
            conn_matrices[band_name] = np.full((num_channels, num_channels), np.nan)

    print("Connectivity calculation complete.")
    return conn_matrices


if __name__ == '__main__':
    print("Testing Connectivity Processor (v3)")
    logbook = load_master_logbook() # Load from parent directory

    if logbook is not None:
        test_row = logbook.dropna(
            subset=['Dream_content', 'Dream_emotion']
        ).iloc[0]

        print(f"\nAttempting to process first valid dream from: {test_row['Online_id']}")

        if not os.path.exists(MAT_FILE_DIR):
             print(f"\nERROR: EEG data folder not found at '{MAT_FILE_DIR}'")
        else:
            connectivity_results = calculate_connectivity(test_row)

            if connectivity_results:
                print("\nSuccessfully calculated connectivity matrices!")
                all_valid = True
                for band, matrix in connectivity_results.items():
                    print(f"\n{band.capitalize()} Band ({CONN_METHOD}):")
                    if np.isnan(matrix).all():
                         print("(Calculation failed or resulted in NaNs)")
                         all_valid = False
                    elif matrix.shape != (6, 6): # Add shape check
                         print(f"(Incorrect shape: {matrix.shape})")
                         all_valid = False
                    else:
                         # Now it should be 6x6
                         print(np.round(matrix, 3))

                # Plot alpha band if valid
                if all_valid:
                    alpha_matrix = connectivity_results.get('alpha')
                    # Check shape explicitly before plotting
                    if alpha_matrix is not None and alpha_matrix.shape == (6, 6) and not np.isnan(alpha_matrix).all():
                        try:
                            plot_matrix = np.array(alpha_matrix)
                            np.fill_diagonal(plot_matrix, 0) # Zero diagonal for plotting

                            plt.figure(figsize=(6, 5))
                            im = plt.imshow(plot_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1) # Coherence range [0, 1]
                            plt.colorbar(im, label=f'Alpha {CONN_METHOD.capitalize()}')
                            plt.title(f"Alpha Connectivity ({CONN_METHOD}) for {test_row['Online_id']}")
                            plt.xticks(ticks=np.arange(6), labels=[f'Ch{i+1}' for i in range(6)])
                            plt.yticks(ticks=np.arange(6), labels=[f'Ch{i+1}' for i in range(6)])
                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            print(f"Could not plot matrix: {e}")
                    elif alpha_matrix is not None:
                         print(f"Could not plot: Alpha matrix shape is {alpha_matrix.shape}, expected (6, 6).")

            else:
                print("\nCould not calculate connectivity for test row. Check errors above.")