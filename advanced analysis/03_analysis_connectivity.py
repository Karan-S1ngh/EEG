import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

CONNECTIVITY_FEATURES_FILE = os.path.join(project_dir, 'connectivity_features.csv')
PLOT_SAVE_PATH = os.path.join(project_dir, 'saved_plots')
PLOT_NAME_SIG = 'significant_connectivity_boxplot.png'
N_TOP_SIGNIFICANT = 5
CONN_METHOD = 'coh'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def analyze_connectivity_features():
    """
    Loads connectivity features and performs ANOVA to find differences
    between emotion groups. V2 fixes NameError.
    """
    # 1. Load the connectivity feature dataset
    try:
        df = pd.read_csv(CONNECTIVITY_FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Connectivity features file not found: '{CONNECTIVITY_FEATURES_FILE}'")
        print("Please run build_connectivity_features.py first.")
        return

    print(f"Loaded connectivity dataset with {df.shape[0]} samples and {df.shape[1]} columns.")

    # 2. Prepare data for analysis
    df_clean = df.dropna(subset=['Emotion_Label'])
    conn_feature_columns = [col for col in df_clean.columns if '_Ch' in col]

    if not conn_feature_columns:
        print("Error: No connectivity feature columns found.")
        return

    print(f"Found {len(conn_feature_columns)} connectivity features to analyze.")

    emotion_counts = df_clean['Emotion_Label'].value_counts()
    print("\nEmotion Class distribution")
    print(emotion_counts)
    valid_groups = emotion_counts[emotion_counts >= 3].index.tolist()
    df_analysis = df_clean[df_clean['Emotion_Label'].isin(valid_groups)]

    if len(valid_groups) < 2:
        print("\nError: Need at least 2 emotion groups with sufficient samples. Stopping.")
        return

    print(f"Using {df_analysis.shape[0]} samples across {len(valid_groups)} groups for ANOVA.")


    # 3. Perform Statistical Analysis (ANOVA)
    print("\nStatistical Analysis (ANOVA)")
    print("Comparing connectivity features across emotion groups...")
    print("p-value < 0.05 suggests a statistically significant difference.")

    significant_results = {}
    for feature in conn_feature_columns:
        groups_data = []
        for group_name in valid_groups:
             group_feature_data = df_analysis[df_analysis['Emotion_Label'] == group_name][feature].dropna()
             if not group_feature_data.empty:
                  groups_data.append(group_feature_data)

        if len(groups_data) >= 2:
            try:
                f_val, p_val = stats.f_oneway(*groups_data)
                if p_val < 0.05:
                    print(f"  - SIGNIFICANT: {feature}: p-value = {p_val:.4f}")
                    significant_results[feature] = p_val
            except Exception:
                 pass # Ignore errors for individual features


    # 4. Report Summary and Visualize Top Findings
    if not significant_results:
        print("\nNo statistically significant differences found.")
    else:
        print(f"\nFound {len(significant_results)} connectivity features showing significant differences.")
        sorted_significant = sorted(significant_results.items(), key=lambda item: item[1])

        print(f"\nTop {min(N_TOP_SIGNIFICANT, len(sorted_significant))} most significant features:")
        for feature, p_val in sorted_significant[:N_TOP_SIGNIFICANT]:
            print(f"  - {feature} (p = {p_val:.4f})")

        top_feature_names = [f for f, p in sorted_significant[:N_TOP_SIGNIFICANT]]

        if top_feature_names:
            top_feature = top_feature_names[0] # Plot only the single most significant
            print(f"\nGenerating boxplot for the top significant feature: {top_feature}")
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df_analysis, x='Emotion_Label', y=top_feature, order=valid_groups)
            plt.title(f'Connectivity Difference: {top_feature}')
            plt.xlabel('Reported Dream Emotion')
            plt.ylabel(f'{CONN_METHOD.capitalize()} Value')
            plt.tight_layout()

            if not os.path.exists(PLOT_SAVE_PATH):
                os.makedirs(PLOT_SAVE_PATH)
            plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME_SIG)
            try:
                plt.savefig(plot_filepath)
                plt.close()
                print(f"Successfully saved boxplot to '{plot_filepath}'")
            except Exception as e:
                print(f"Error saving boxplot: {e}")
                plt.show()


if __name__ == '__main__':
    analyze_connectivity_features()