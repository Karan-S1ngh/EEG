import os
import sys
import pandas as pd
import numpy as np
import re # Regular expressions for cleaning text
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

try:
    from nltk.corpus import stopwords
    stop_words = list(stopwords.words('english'))
    # Add custom stopwords used previously
    stop_words.extend(['dream', 'dreamed', 'thinking', 'think', 'like', 'went', 'one', 'get'])
except LookupError:
    print("Error: NLTK stopwords not found. Run 'python -m nltk.downloader stopwords' in your main Python environment.")
    sys.exit(1)

LOGBOOK_FILE = os.path.join(project_dir, 'master_logbook.csv')
OUTPUT_FILE = os.path.join(project_dir, 'topic_scores.csv') # Saves in main dir
N_TOPICS = 10 # Must match the number of topics used in previous analysis

warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Basic text cleaning: lowercase, remove punctuation/numbers, remove stopwords."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def save_topic_scores():
    """
    Loads dream text, runs LDA, calculates topic distributions for every dream,
    and saves the scores to a new CSV file.
    """
    # 1. Load the master logbook
    try:
        df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError:
        print(f"Error: Logbook file not found: '{LOGBOOK_FILE}'")
        return

    # 2. Prepare Data and Add Emotion Labels
    df_clean = df.dropna(subset=['Dream_content', 'Dream_emotion']).copy()

    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion: return 'Negative'
        if 'happy' in emotion: return 'Positive'
        if 'calm' in emotion: return 'Calm'
        return None

    df_clean.loc[:, 'Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion)
    df_analysis = df_clean.dropna(subset=['Emotion_Label'])

    # 3. Preprocess Text
    preprocessed_docs = df_analysis['Dream_content'].apply(preprocess_text)
    valid_indices = preprocessed_docs[preprocessed_docs.str.len() > 0].index
    preprocessed_docs = preprocessed_docs.loc[valid_indices]
    df_analysis = df_analysis.loc[valid_indices] # Match metadata

    if len(preprocessed_docs) < N_TOPICS:
        print("Error: Not enough valid dream documents.")
        return

    print(f"Using {len(preprocessed_docs)} valid dream documents for topic modeling.")

    # 4. Create Document-Term Matrix (Need to fit on full corpus)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=stop_words)
    dtm = vectorizer.fit_transform(preprocessed_docs)

    # 5. Run LDA (Fit the same model as before)
    print(f"Running LDA to find {N_TOPICS} topics...")
    lda = LatentDirichletAllocation(n_components=N_TOPICS, max_iter=10,
                                    learning_method='online', learning_offset=50.,
                                    random_state=42)
    lda.fit(dtm)

    # 6. Transform and Save Topic Scores
    print("Transforming documents to get topic probabilities...")
    topic_distribution = lda.transform(dtm) # Topic probability for each dream
    topic_col_names = [f'Topic_{i+1}_Prob' for i in range(N_TOPICS)]
    topic_df = pd.DataFrame(topic_distribution, columns=topic_col_names, index=df_analysis.index)

    # Merge Topic Scores with IDs and Labels
    cols_to_keep = ['Online_id', 'Emotion_Label', 'Dream_content', 'REM_period_start_time']
    final_df = pd.concat([df_analysis[cols_to_keep], topic_df], axis=1).dropna(subset=topic_col_names)

    # 7. Save to CSV
    try:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved topic scores to '{OUTPUT_FILE}'!")
        print(f"Dataset shape: {final_df.shape}")
        print("\nPreview:")
        print(final_df.head())
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == '__main__':
    save_topic_scores()