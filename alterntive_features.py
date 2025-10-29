import os
import sys
import scipy.io
import numpy as np
import pandas as pd
import mne
import warnings
from tqdm import tqdm
from pandas.errors import SettingWithCopyWarning

try:
    import antropy as ant # Use only antropy
except ImportError:
    print("Error: 'antropy' library not found. Please install it: pip install antropy")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir 
MAT_FILE_DIR = os.path.join(project_dir, 'eeg_mat_files')
OUTPUT_FILE = os.path.join(project_dir, 'entropy_features.csv')

if project_dir not in sys.path:
     sys.path.append(project_dir)
parent_dir = os.path.dirname(project_dir)
if parent_dir not in sys.path:
     sys.path.append(parent_dir)
try:
    from data_loader import load_master_logbook
except ImportError:
    try:
        from data_loader import load_master_logbook
        print("INFO: Imported data_loader from current directory.")
    except ImportError:
        print(f"Error: Could not import load_master_logbook from {project_dir} or parent.")
        sys.exit(1)

print(f"DEBUG: Script directory: {script_dir}")
print(f"DEBUG: Project directory now set to: {project_dir}")
print(f"DEBUG: Expecting .mat files in: {MAT_FILE_DIR}")
if not os.path.exists(MAT_FILE_DIR):
    print(f"FATAL ERROR: The directory '{MAT_FILE_DIR}' does not exist! Please check the path and folder name.")
    sys.exit(1)
else:
    print(f"INFO: Found MAT file directory: '{MAT_FILE_DIR}'")


SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'
NUM_CHANNELS = 6
ENTROPY_ORDER = 3
ENTROPY_DELAY = 1

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def calculate_entropy_features(eeg_segment, srate=SAMPLING_RATE, participant_id="Unknown"):
    """Calculates Permutation Entropy features for each channel."""
    num_channels = eeg_segment.shape[0]
    if num_channels != NUM_CHANNELS: return None
    features = {}
    calculation_successful = True
    for i in range(num_channels):
        channel_data = eeg_segment[i, :]
        ch_name = f'Ch{i+1}'
        try:
            perm_entropy = ant.perm_entropy(channel_data, order=ENTROPY_ORDER, delay=ENTROPY_DELAY, normalize=True)
            if np.isnan(perm_entropy):
                 features[f'{ch_name}_PermutationEntropy'] = np.nan
                 calculation_successful = False
            else:
                 features[f'{ch_name}_PermutationEntropy'] = perm_entropy
        except Exception as e:
            features[f'{ch_name}_PermutationEntropy'] = np.nan
            calculation_successful = False
    if not calculation_successful: return None
    else: return features

def create_entropy_feature_dataset():
    """Loops through dreams, calculates entropy features, and saves."""
    logbook = load_master_logbook()
    if logbook is None: return

    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion: return 'Negative'
        if 'happy' in emotion: return 'Positive'
        if 'calm' in emotion: return 'Calm'
        return None
    logbook['Emotion_Label'] = logbook['Dream_emotion'].apply(map_emotion)

    all_features_list = []
    print(f"\nStarting Entropy Feature Extraction")
    processed_count = 0
    skip_reasons = {'file': 0, 'short': 0, 'load': 0, 'channels': 0, 'filter': 0, 'calc': 0}

    try:
        mat_files_list = os.listdir(MAT_FILE_DIR)
        print(f"DEBUG: Found {len(mat_files_list)} files/folders in MAT directory.")
    except Exception as e:
        print(f"FATAL ERROR: Could not list files in '{MAT_FILE_DIR}'. Error: {e}")
        return

    for index, row in tqdm(logbook.iterrows(), total=logbook.shape[0]):
        participant_id = row['Online_id']
        emotion_label = row['Emotion_Label']
        video_type = row['Video_type']

        mat_file_name = None; found_file = False
        for file in mat_files_list:
            # Make matching case-insensitive just in case
            if participant_id.lower() in file.lower() and file.lower().endswith('.mat'):
                mat_file_name = file; found_file = True; break

        if not found_file: skip_reasons['file'] += 1; continue

        mat_file_path = os.path.join(MAT_FILE_DIR, mat_file_name)
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            eeg_segment = mat_data[EEG_VARIABLE_NAME]
            segment_length = eeg_segment.shape[1]
            required_length = ENTROPY_ORDER * ENTROPY_DELAY + 10 # Check length needed for entropy calc
            if segment_length < required_length: skip_reasons['short'] += 1; continue
        except Exception as e: skip_reasons['load'] += 1; continue

        num_ch = eeg_segment.shape[0]
        if num_ch != NUM_CHANNELS: skip_reasons['channels'] += 1; continue

        ch_names = [f'Ch{i+1}' for i in range(num_ch)]
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=['eeg']*num_ch)
        raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False) # uV to V
        try:
            raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
            filtered_eeg_segment = raw.get_data()
        except Exception as e: skip_reasons['filter'] += 1; continue

        entropy_features_result = calculate_entropy_features(filtered_eeg_segment, srate=SAMPLING_RATE, participant_id=participant_id)

        if entropy_features_result is None:
            skip_reasons['calc'] += 1
            continue

        entropy_features = entropy_features_result
        entropy_features['Online_id'] = participant_id
        entropy_features['Emotion_Label'] = emotion_label
        entropy_features['Video_type'] = video_type
        all_features_list.append(entropy_features)
        processed_count += 1

    print(f"\nSuccessfully processed {processed_count} dreams out of {logbook.shape[0]} attempted.")
    print("Skip reasons:")
    print(f"  - File not found: {skip_reasons['file']}")
    print(f"  - Segment too short: {skip_reasons['short']}")
    print(f"  - Error loading .mat: {skip_reasons['load']}")
    print(f"  - Incorrect channels: {skip_reasons['channels']}")
    print(f"  - Filtering failed: {skip_reasons['filter']}")
    print(f"  - Calculation failed/NaN: {skip_reasons['calc']}")

    if all_features_list:
        features_df = pd.DataFrame(all_features_list)
        try:
            features_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSuccessfully saved entropy features to '{OUTPUT_FILE}'!")
            print("\nPreview:")
            print(features_df.head())
            print(f"Dataset shape: {features_df.shape}")
        except Exception as e:
            print(f"Error saving features to CSV: {e}")
    elif logbook.shape[0] > 0:
         print("\nNo entropy features were successfully extracted. Check detailed skip reasons above.")


if __name__ == '__main__':
    create_entropy_feature_dataset()