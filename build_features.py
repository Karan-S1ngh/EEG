import pandas as pd
from data_loader import load_master_logbook
from eeg_processor import extract_eeg_features
from tqdm import tqdm

def create_feature_dataset():
    """
    Runs the full EEG extraction pipeline on all dreams and saves the
    results to a new CSV file.
    """
    # 1. Load the master logbook
    logbook = load_master_logbook()
    
    if logbook is None:
        print("Error: Could not load master logbook. Exiting.")
        return

    all_features_list = []
    print(f"\nStarting feature extraction for all {logbook.shape[0]} dream reports")
    
    # 2. Loop through every row in the logbook
    # tqdm will show a progress bar
    for index, row in tqdm(logbook.iterrows(), total=logbook.shape[0]):
        # 3. Extract features for each row
        features = extract_eeg_features(row)
        
        # 4. If successful, add them to our list
        if features is not None:
            all_features_list.append(features)

    print(f"\nSuccessfully extracted features for {len(all_features_list)} out of {logbook.shape[0]} dreams.")

    # 5. Convert the list of dictionaries into a pandas DataFrame
    features_df = pd.DataFrame(all_features_list)
    
    # 6. Save the final, clean dataset to a new CSV file
    output_filename = 'dream_features.csv'
    try:
        features_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved all features to '{output_filename}'!")
        
        print("\nHere's a preview of your final dataset:")
        print(features_df.head())
        
    except Exception as e:
        print(f"Error saving features to CSV: {e}")


if __name__ == '__main__':
    create_feature_dataset()