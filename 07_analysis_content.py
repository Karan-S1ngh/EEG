import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pandas.errors import SettingWithCopyWarning

FEATURES_FILE = 'dream_features.csv'
LOGBOOK_FILE = 'master_logbook.csv'
PLOT_SAVE_PATH = 'saved_plots'
PLOT_NAME = 'content_correlation_heatmap.png'
N_TOP_WORDS = 20 # How many top words to analyze

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def analyze_content_correlation():
    """
    Analyzes the correlation between dream content keywords (TF-IDF)
    and EEG features. Saves a heatmap of the correlations.
    """
    # 1. Load BOTH datasets
    try:
        eeg_df = pd.read_csv(FEATURES_FILE)
        logbook_df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError as e:
        print(f"Error: Missing data file: {e}")
        print("Please run data_loader.py and build_features.py first.")
        return

    print(f"Loaded {eeg_df.shape[0]} EEG feature sets.")
    print(f"Loaded {logbook_df.shape[0]} logbook entries.")

    # 2. Combine the datasets (same logic as multimodal)
    logbook_clean = logbook_df.dropna(subset=['Dream_content', 'Dream_emotion'])
    logbook_clean = logbook_clean.reset_index().rename(columns={'index': 'dream_idx'})
    eeg_df = eeg_df.reset_index().rename(columns={'index': 'dream_idx'})

    df_merged = pd.merge(eeg_df, logbook_clean[['dream_idx', 'Dream_content']],
                         on='dream_idx', how='inner')

    print(f"Merged dataset shape for content analysis: {df_merged.shape}")
    df_analysis = df_merged.dropna(subset=['Dream_content']) # Ensure text is present

    if df_analysis.shape[0] < 10:
        print("Error: Not enough data with both EEG features and dream text.")
        return

    # 3. Extract Keywords using TF-IDF
    print("\nExtracting keywords using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df_analysis['Dream_content'])

    # Get feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame with TF-IDF scores for each word
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df_analysis.index)

    # Find the top N words based on their average TF-IDF score across all dreams
    top_words = tfidf_df.mean().sort_values(ascending=False).head(N_TOP_WORDS).index.tolist()
    print(f"Top {N_TOP_WORDS} keywords found: {top_words}")

    # Keep only the TF-IDF scores for these top words
    top_tfidf_df = tfidf_df[top_words]

    # 4. Prepare EEG features
    eeg_feature_columns = [col for col in df_analysis.columns if col.endswith('_power')]
    eeg_features_df = df_analysis[eeg_feature_columns]

    # 5. Combine Top Keywords and EEG Features
    combined_df = pd.concat([top_tfidf_df, eeg_features_df], axis=1)

    # 6. Calculate the Correlation Matrix
    print("\nCalculating correlations between keywords and EEG features...")
    correlation_matrix = combined_df.corr()

    # We only care about the correlations *between* words and EEG features
    # Select the block of the matrix showing Word <-> EEG correlations
    word_eeg_corr = correlation_matrix.loc[top_words, eeg_feature_columns]

    # 7. Visualize and Save the Heatmap
    print("Generating heatmap")
    plt.figure(figsize=(18, 10)) # Adjust size as needed
    sns.heatmap(word_eeg_corr, annot=False, cmap='coolwarm', center=0) 
    plt.title(f'Correlation between Top {N_TOP_WORDS} Dream Keywords (TF-IDF) and EEG Band Power')
    plt.xlabel('EEG Features (Channel_Band_Power)')
    plt.ylabel('Top Dream Keywords')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)
    plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME)
    plt.savefig(plot_filepath)
    plt.close()

    print(f"\nSuccessfully saved correlation heatmap to '{plot_filepath}'")

    # Flatten the matrix and find the largest absolute correlations
    corr_unstacked = word_eeg_corr.unstack().sort_values(key=abs, ascending=False)
    print("\nTop 5 Strongest Correlations (Positive or Negative):")
    print(corr_unstacked.head(5))


if __name__ == '__main__':
    analyze_content_correlation()