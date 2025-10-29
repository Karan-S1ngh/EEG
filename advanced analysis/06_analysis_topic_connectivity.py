import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add main project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Configuration
TOPIC_SCORES_FILE = os.path.join(project_dir, 'topic_scores.csv')
CONNECTIVITY_FEATURES_FILE = os.path.join(project_dir, 'connectivity_features.csv')
PLOT_SAVE_PATH = os.path.join(project_dir, 'saved_plots')
PLOT_NAME = 'topic_connectivity_correlation.png'
N_TOP_CORRELATIONS = 10 # How many top correlations to report

# Suppress warnings
warnings.filterwarnings('ignore')


def analyze_topic_connectivity():
    """
    Loads topic scores and connectivity features, merges them, and calculates
    the correlation matrix between dream themes and brain synchronization.
    """
    # 1. Load Topic Scores and Connectivity Features
    try:
        df_topics = pd.read_csv(TOPIC_SCORES_FILE)
        df_conn = pd.read_csv(CONNECTIVITY_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"Error: Missing feature file: {e}")
        print("Please ensure 'topic_scores.csv' and 'connectivity_features.csv' exist in the main directory.")
        return

    print(f"Loaded topic scores shape: {df_topics.shape}")
    print(f"Loaded connectivity features shape: {df_conn.shape}")

    # 2. Prepare for Merge
    # We must ensure both DataFrames have the same indices and length after cleaning
    # Since both were derived from the same logbook, a simple index reset merge should work.
    df_topics = df_topics.drop(columns=['Emotion_Label', 'Video_type', 'REM_period_start_time'], errors='ignore')
    df_conn = df_conn.drop(columns=['Emotion_Label', 'Video_type'], errors='ignore')
    
    # Use Online_id as a key, dropping duplicates if necessary (though features should be unique)
    # Resetting index to ensure a clean merge if data rows were skipped earlier
    df_topics = df_topics.reset_index(drop=True)
    df_conn = df_conn.reset_index(drop=True)
    
    # Perform Merge on Online_id
    df_merged = pd.merge(df_topics, df_conn, on='Online_id', how='inner')

    print(f"Merged dataset shape for correlation: {df_merged.shape}")
    
    # 3. Define Feature Sets
    topic_col_names = [col for col in df_merged.columns if col.endswith('_Prob')]
    conn_col_names = [col for col in df_merged.columns if '_Ch' in col]

    if not topic_col_names or not conn_col_names:
        print("Error: Could not identify Topic or Connectivity columns.")
        return

    print(f"Analyzing {len(topic_col_names)} Topics and {len(conn_col_names)} Connectivity Features.")


    # 4. Calculate the Correlation Matrix
    # Select only the features we need for the final matrix calculation
    df_features = df_merged[topic_col_names + conn_col_names]
    
    # Calculate full correlation matrix
    correlation_matrix = df_features.corr()

    # Isolate the Topic-Connectivity block (Topic rows vs. Connectivity columns)
    topic_conn_corr = correlation_matrix.loc[topic_col_names, conn_col_names]

    # 5. Report Strongest Correlations
    corr_unstacked = topic_conn_corr.unstack().sort_values(key=abs, ascending=False)
    
    print(f"\nTop {N_TOP_CORRELATIONS} Strongest Correlations (Topics <-> Connectivity)")
    for i, ((conn_feature, topic_feature), corr_val) in enumerate(corr_unstacked.head(N_TOP_CORRELATIONS).items()):
        print(f"  {i+1}. {topic_feature:<15} <-> {conn_feature:<20} | R = {corr_val:.4f}")
    
    print("\nInterpretation: R > 0 suggests a topic's prevalence increases when that connection is stronger.")


    # 6. Visualize and Save the Heatmap
    print("\nGenerating heatmap...")
    plt.figure(figsize=(15, 10))
    # Use different vmin/vmax for clearer visualization of the R values
    sns.heatmap(topic_conn_corr, annot=False, cmap='coolwarm', vmin=-0.25, vmax=0.25, center=0) 
    plt.title('Correlation between Dream Topic Prevalence and Brain Connectivity (R)')
    plt.xlabel('Connectivity Features (Band_ChannelPair)')
    plt.ylabel('Dream Topic Probability')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOT_SAVE_PATH): os.makedirs(PLOT_SAVE_PATH)
    plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME)
    try:
        plt.savefig(plot_filepath)
        plt.close()
        print(f"Successfully saved Topic-Connectivity heatmap to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        plt.show()


if __name__ == '__main__':
    analyze_topic_connectivity()