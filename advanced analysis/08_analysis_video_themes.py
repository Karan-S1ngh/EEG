import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

project_dir = os.getcwd() 
if project_dir not in sys.path:
     sys.path.append(project_dir)

TOPIC_SCORES_FILE = os.path.join(project_dir, 'topic_scores.csv')
LOGBOOK_FILE = os.path.join(project_dir, 'master_logbook.csv') # Source of Video_type
PLOT_SAVE_PATH = os.path.join(project_dir, 'saved_plots')
PLOT_NAME_THEMES = 'topic_video_comparison_chart.png'

TOPIC_INTERPRETATIONS = {
    'Topic_1_Prob': 'School/Holiday/Activity',
    'Topic_2_Prob': 'Family/Social Talk',
    'Topic_3_Prob': 'Financial/Goal Setting',
    'Topic_4_Prob': 'Action/Conflict',
    'Topic_5_Prob': 'School/Home Environment',
    'Topic_6_Prob': 'Fighting/Vague Scenes',
    'Topic_7_Prob': 'Experiment/Cognition',
    'Topic_8_Prob': 'Memory/Objects',
    'Topic_9_Prob': 'Games/Competition',
    'Topic_10_Prob': 'Leisure/Food'
}

warnings.filterwarnings('ignore')


def analyze_video_themes():
    """
    Loads topic scores and compares average topic prevalence between
    Positive and Negative video groups using T-Tests by merging with logbook.
    """
    # 1. Load Topic Scores and Logbook
    try:
        df_topics = pd.read_csv(TOPIC_SCORES_FILE)
        # Load logbook just to get the cleaner Video_type column
        df_logbook = pd.read_csv(LOGBOOK_FILE) 
    except FileNotFoundError as e:
        print(f"Error: Missing feature file: {e}. Ensure files exist.")
        return

    print(f"Loaded topic scores shape: {df_topics.shape}")

    # 2. Prepare Merge: Use Online_id and REM_period_start_time as keys
    # Clean up the key columns (CRUCIAL for string matching)
    df_topics['Online_id'] = df_topics['Online_id'].astype(str).str.strip()
    df_topics['REM_period_start_time'] = df_topics['REM_period_start_time'].astype(str).str.strip()
    
    df_logbook['Online_id'] = df_logbook['Online_id'].astype(str).str.strip()
    df_logbook['REM_period_start_time'] = df_logbook['REM_period_start_time'].astype(str).str.strip()

    # Select only the Video_type column from the logbook
    df_video_key = df_logbook[['Online_id', 'REM_period_start_time', 'Video_type']].copy()
    df_video_key = df_video_key.dropna(subset=['Video_type'])


    # Merge the Topic Scores with the clean Video_type column
    df_analysis = pd.merge(df_topics, df_video_key, 
                           on=['Online_id', 'REM_period_start_time'], 
                           how='inner')
    
    # Check the resulting shape
    print(f"Merged analysis dataset shape: {df_analysis.shape}")
    
    # 3. Filter for only Positive and Negative Video Groups
    df_model = df_analysis[df_analysis['Video_type'].isin(['Positive', 'Negative'])].copy()
    
    if df_model.shape[0] < 50:
        print("Error: Insufficient data after filtering for Pos/Neg videos. Stopping.")
        return

    print(f"Analyzing {df_model.shape[0]} dreams following Positive or Negative videos.")

    # 4. Define Feature Sets
    topic_col_names = [col for col in df_model.columns if col.endswith('_Prob')]
    
    # 5. Compare Averages and Run T-Tests
    positive_dreams = df_model[df_model['Video_type'] == 'Positive'][topic_col_names]
    negative_dreams = df_model[df_model['Video_type'] == 'Negative'][topic_col_names]

    comparison_results = []
    print("\nTopic Prevalence Comparison (Negative Video vs. Positive Video)")

    for topic in topic_col_names:
        try:
            stat, p_val = stats.ttest_ind(negative_dreams[topic].dropna(), positive_dreams[topic].dropna(), equal_var=False)
            diff = negative_dreams[topic].mean() - positive_dreams[topic].mean()
            
            comparison_results.append({'Topic': topic, 'P_Value': p_val, 'Mean_Diff': diff})
            
            if p_val < 0.05:
                topic_name = TOPIC_INTERPRETATIONS.get(topic, topic)
                print(f"  - SIGNIFICANT: {topic_name}: P={p_val:.4f} (Negative > Positive by {diff:.3f})")

        except Exception as e:
            pass

    results_df = pd.DataFrame(comparison_results).sort_values(by='P_Value')
    
    # 6. Visualize the Most Significant Topic Differences
    significant_topics = results_df[results_df['P_Value'] < 0.05]
    
    if significant_topics.empty:
        print("\nConclusion: No statistically significant thematic differences found.")
        return

    print(f"\nConclusion: Found {len(significant_topics)} topics significantly affected by video valence.")
    
    # Create plot data
    plot_data = significant_topics.set_index('Topic').sort_values(by='Mean_Diff', ascending=False)
    
    def get_display_name(topic):
         return TOPIC_INTERPRETATIONS.get(topic, topic.replace('_Prob', ''))
         
    plot_data.index = plot_data.index.map(get_display_name)
    
    # Create Plot (Bar Chart showing the difference)
    plt.figure(figsize=(10, 7))
    plot_data['Mean_Diff'].plot(kind='barh', color=plot_data['Mean_Diff'].apply(lambda x: 'darkred' if x > 0 else 'darkblue'))
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Difference in Topic Prevalence (Negative Video Dreams - Positive Video Dreams)', fontsize=14)
    plt.xlabel('Mean Topic Probability Difference')
    plt.ylabel('Significant Dream Topic')
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(PLOT_SAVE_PATH): os.makedirs(PLOT_SAVE_PATH)
    plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME_THEMES)
    try:
        plt.savefig(plot_filepath)
        plt.close()
        print(f"\nSuccessfully saved comparative theme chart to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving chart: {e}")


if __name__ == '__main__':
    analyze_video_themes()