import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import warnings

FEATURES_FILE = 'dream_features.csv'
PLOT_SAVE_PATH = 'saved_plots'
PLOT_NAME = 'video_impact_plot.png'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def analyze_video_impact():
    """
    Loads the features, analyzes the impact of video type on EEG features,
    and saves a plot. Uses all 30 EEG features.
    """
    # 1. Load the clean 30-feature dataset
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Features file not found: '{FEATURES_FILE}'")
        print("Please run build_features.py first.")
        return

    print(f"Loaded feature dataset with {df.shape[0]} samples.")

    # 2. Prepare data for this analysis
    # Drop rows that don't have a Video_type
    df_clean = df.dropna(subset=['Video_type'])

    # Check the counts
    video_counts = df_clean['Video_type'].value_counts()
    print("\nVideo type counts (after dropping NaNs):")
    print(video_counts)

    # Filter to keep only the main groups (adjust if your labels are different)
    valid_groups = ['Positive', 'Negative', 'Neutral']
    df_model = df_clean[df_clean['Video_type'].isin(valid_groups)]

    if df_model.shape[0] < 10 or len(df_model['Video_type'].unique()) < 2:
        print("\nError: Not enough data or groups after cleaning video labels. Stopping.")
        print("Need at least 2 different video types with enough samples.")
        return

    print(f"Cleaned data for video analysis: {df_model.shape[0]} samples across {len(df_model['Video_type'].unique())} groups.")

    # 3. Perform Statistical Analysis (ANOVA)
    # Find all EEG feature columns
    feature_columns = [col for col in df_model.columns if col.endswith('_power')]

    print("\nStatistical Analysis (ANOVA)")
    print("Comparing EEG features across video groups...")
    print("p-value < 0.05 suggests a statistically significant difference.")

    significant_features = 0
    for feature in feature_columns:
        # Collect the feature values for each group present in the data
        groups_data = []
        present_groups = df_model['Video_type'].unique()
        for group_name in present_groups:
             groups_data.append(df_model[df_model['Video_type'] == group_name][feature])

        # Run the ANOVA test only if we have at least 2 groups
        if len(groups_data) >= 2:
            try:
                f_val, p_val = stats.f_oneway(*groups_data)
                if p_val < 0.05:
                    print(f"  - SIGNIFICANT: {feature}: p-value = {p_val:.4f}")
                    significant_features += 1
                # else: # Optionally print non-significant ones too
                #     print(f"  - {feature}: p-value = {p_val:.4f}")
            except ValueError as e:
                # Handle cases where a group might have only one sample
                print(f"  - Could not run ANOVA for {feature}: {e}")
        else:
             print(f"  - Skipping ANOVA for {feature}: Not enough groups.")


    if significant_features == 0:
        print("\nNo statistically significant differences found in EEG features between video groups.")
    else:
        print(f"\nFound {significant_features} EEG features showing significant differences between video groups.")


    # 4. Group data by video type and calculate mean for *all* features
    # We calculate means even if not significant, for plotting
    grouped_data_all_features = df_model.groupby('Video_type')[feature_columns].mean()

    # Calculate an overall average power for each band across channels for a simpler plot
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    grouped_data_avg_bands = pd.DataFrame(index=grouped_data_all_features.index)

    for band in bands:
        band_cols = [col for col in feature_columns if f'_{band}_power' in col]
        grouped_data_avg_bands[f'{band}_power'] = grouped_data_all_features[band_cols].mean(axis=1)

    print("\nAverage band power (across channels) per video group:")
    print(grouped_data_avg_bands)

    # 5. Create and save the plot
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)

    plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME)

    grouped_data_avg_bands.plot(kind='bar', figsize=(10, 6))
    plt.title('Average EEG Band Power by Pre-Sleep Video Type')
    plt.ylabel('Average Log(Power)')
    plt.xlabel('Video Type')
    plt.xticks(rotation=0)
    plt.legend(title='EEG Bands')
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.close() # Close the plot to free memory

    print(f"\nSuccessfully saved analysis plot to '{plot_filepath}'")


if __name__ == '__main__':
    analyze_video_impact()