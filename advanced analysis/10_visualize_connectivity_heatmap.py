import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

project_dir = os.getcwd()
if project_dir not in sys.path:
    sys.path.append(project_dir)

CONNECTIVITY_FEATURES_FILE = os.path.join(project_dir, 'connectivity_features.csv')
PLOT_SAVE_PATH = os.path.join(project_dir, 'saved_plots')
PLOT_NAME = 'connectivity_heatmap_alpha.png' # New name
CONN_METHOD = 'coh'
BAND_TO_PLOT = 'alpha' # The band we found was most significant

ASSUMED_CHANNEL_NAMES = ['Fz', 'Cz', 'Pz', 'Oz', 'C3', 'C4']
NUM_CHANNELS = len(ASSUMED_CHANNEL_NAMES)

warnings.filterwarnings('ignore')

def visualize_connectivity_heatmap():
    """
    Loads connectivity features, calculates the average matrix for a specific band,
    and plots it as a standard heatmap.
    """
    # 1. Load data
    try:
        df = pd.read_csv(CONNECTIVITY_FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: File not found: '{CONNECTIVITY_FEATURES_FILE}'")
        return
    print(f"Loaded connectivity dataset with {df.shape[0]} samples.")

    # 2. Get columns for the chosen band
    band_feature_columns = [col for col in df.columns if col.startswith(f'{BAND_TO_PLOT}_Ch')]
    if not band_feature_columns:
        print(f"Error: No features found for band '{BAND_TO_PLOT}'.")
        return
    
    print(f"Analyzing {len(band_feature_columns)} connections for '{BAND_TO_PLOT}' band.")

    # 3. Calculate the average strength for each connection
    avg_strengths = df[band_feature_columns].mean()

    # 4. Rebuild the 6x6 Adjacency Matrix
    adj_matrix = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
    for feature_name, avg_strength in avg_strengths.items():
        try:
            parts = feature_name.split('_'); idx1 = int(parts[1][2:])-1; idx2 = int(parts[2][2:])-1
            if 0 <= idx1 < NUM_CHANNELS and 0 <= idx2 < NUM_CHANNELS:
                adj_matrix[idx1, idx2] = avg_strength
                adj_matrix[idx2, idx1] = avg_strength # Make symmetric
        except Exception:
            pass # Ignore parsing errors

    # 5. Plot the Heatmap
    print(f"Generating heatmap for average {BAND_TO_PLOT} connectivity...")
    try:
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            adj_matrix,
            xticklabels=ASSUMED_CHANNEL_NAMES,
            yticklabels=ASSUMED_CHANNEL_NAMES,
            annot=True,     # Show the numbers in the squares
            fmt=".2f",      # Format numbers to 2 decimal places
            cmap='viridis', # Colormap (can change to 'hot', 'coolwarm', etc.)
            linewidths=.5,
            vmin=0, vmax=1   # Set scale from 0 to 1 for coherence
        )
        plt.title(f'Average {BAND_TO_PLOT.capitalize()} Brain Connectivity (Coherence)')
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(PLOT_SAVE_PATH): os.makedirs(PLOT_SAVE_PATH)
        plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME)
        plt.savefig(plot_filepath, dpi=150)
        plt.close()
        print(f"\nSuccessfully saved connectivity heatmap to '{plot_filepath}'")

    except Exception as e:
        print(f"Error generating heatmap: {e}")


if __name__ == '__main__':
    visualize_connectivity_heatmap()