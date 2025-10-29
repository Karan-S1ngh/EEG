import pandas as pd
import os

def load_master_logbook(data_path='.'):
    """
    Loads, cleans, and merges the three main files into a single master DataFrame.
    Uses pd.read_excel because the files are Excel files renamed to .csv.
    """
    # Define file paths
    emotional_ratings_file = os.path.join(data_path, 'Emotional_ratings_excel_files.xlsx')
    status_file = os.path.join(data_path, 'Status_identification_of_each_stage_of_EEG.xlsx')
    video_list_file = os.path.join(data_path, 'Video_list.xlsx')

    # Load the files using pd.read_excel
    try:
        ratings_df = pd.read_excel(emotional_ratings_file, engine='openpyxl')
        status_df = pd.read_excel(status_file, engine='openpyxl')
        video_df = pd.read_excel(video_list_file, engine='openpyxl')
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the CSV files are in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        print("Please make sure you have installed 'openpyxl' by running: pip install openpyxl")
        return None

    # Clean the Emotional Ratings DataFrame
    ratings_df['Online_id'] = ratings_df['Online_id'].ffill()

    # Drop rows that don't have a dream report (i.e., 'Dream_content' is empty)
    ratings_df.dropna(subset=['Dream_content'], inplace=True)
    
    # Merge the DataFrames
    # Merge ratings with status info
    master_df = pd.merge(ratings_df, status_df, on='Online_id', how='left')

    # Merge the result with video info
    master_df = pd.merge(master_df, video_df, on='Online_id', how='left')

    print("Successfully loaded and merged all data files!")
    print(f"Master logbook has {master_df.shape[0]} rows and {master_df.shape[1]} columns.")
    
    return master_df


if __name__ == '__main__':
    # This is a test block to see if the function works when you run this file directly.
    
    master_logbook = load_master_logbook()
    
    if master_logbook is not None:
        print("\nFirst 5 rows of the master logbook")
        print(master_logbook.head())
        
        print("\nInfo about the master logbook")
        master_logbook.info()
        
        # Save the merged DataFrame to a new CSV file
        try:
            master_logbook.to_csv('master_logbook.csv', index=False, encoding='utf-8')
            print("\nSuccessfully saved merged data to 'master_logbook.csv'")
        except Exception as e:
            print(f"\nCould not save master_logbook.csv. Error: {e}")