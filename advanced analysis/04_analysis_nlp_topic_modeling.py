import os
import sys
import pandas as pd
import numpy as np
import re # Regular expressions for cleaning text
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns        
from pandas.errors import SettingWithCopyWarning

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
try:
    from nltk.corpus import stopwords
    stop_words = list(stopwords.words('english'))
    # Add custom words if needed (e.g., words related to the experiment)
    stop_words.extend(['dream', 'dreamed', 'thinking', 'think', 'like', 'one', 'get', 'went'])
except LookupError:
    print("Error: NLTK stopwords not found.")
    print("Please download them: open Python, run 'import nltk', then 'nltk.download(\"stopwords\")'")
    sys.exit(1)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

LOGBOOK_FILE = os.path.join(project_dir, 'master_logbook.csv')
PLOT_SAVE_PATH = os.path.join(project_dir, 'saved_plots') # To save plots
PLOT_NAME_TOPIC_EMOTION = 'topic_emotion_heatmap.png'
N_TOPICS = 10 # How many topics to look for (can be tuned)
N_TOP_WORDS_PER_TOPIC = 10 # How many words to display for each topic

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning) # For pandas
warnings.filterwarnings('ignore', category=DeprecationWarning) # For scikit-learn


def preprocess_text(text):
    """Basic text cleaning: lowercase, remove punctuation/numbers, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2] # Remove stopwords and short words
    return " ".join(words)

def display_topics(model, feature_names, n_top_words):
    """Prints the top words for each topic found by LDA."""
    print(f"\nTop {n_top_words} words per Topic")
    topic_summaries = {}
    for topic_idx, topic in enumerate(model.components_):
        top_feature_indices = topic.argsort()[:-n_top_words - 1:-1]
        feature_names_np = np.array(feature_names)
        top_words = feature_names_np[top_feature_indices]
        topic_name = f"Topic {topic_idx+1}"
        print(f"{topic_name}: {', '.join(top_words)}")
        topic_summaries[topic_name] = top_words # Store for later use
    return topic_summaries

def run_topic_modeling():
    """
    Loads dream text, preprocesses, runs LDA, displays topics,
    and analyzes topic distribution per emotion.
    """
    # 1. Load the master logbook
    try:
        df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError:
        print(f"Error: Logbook file not found: '{LOGBOOK_FILE}'")
        return

    print(f"Loaded master logbook with {df.shape[0]} samples.")

    # 2. Prepare and Preprocess Text Data
    df_clean = df.dropna(subset=['Dream_content', 'Dream_emotion'])

    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion: return 'Negative'
        if 'happy' in emotion: return 'Positive'
        if 'calm' in emotion: return 'Calm'
        return None
    df_clean = df_clean.copy()
    df_clean['Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion)
    df_analysis = df_clean.dropna(subset=['Emotion_Label'])

    print("Preprocessing dream text (cleaning, removing stopwords)...")
    preprocessed_docs = df_analysis['Dream_content'].apply(preprocess_text)

    valid_indices = preprocessed_docs[preprocessed_docs.str.len() > 0].index
    if len(valid_indices) < df_analysis.shape[0]:
         print(f"Warning: Removed {df_analysis.shape[0] - len(valid_indices)} dreams that were empty after preprocessing.")
    preprocessed_docs = preprocessed_docs.loc[valid_indices]
    df_analysis = df_analysis.loc[valid_indices] # Keep corresponding metadata

    if len(preprocessed_docs) < N_TOPICS:
        print("Error: Not enough valid dream documents for LDA.")
        return

    print(f"Using {len(preprocessed_docs)} valid dream documents.")

    # 3. Create Document-Term Matrix
    print("Creating document-term matrix...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=stop_words)
    dtm = vectorizer.fit_transform(preprocessed_docs)
    feature_names = vectorizer.get_feature_names_out()

    # 4. Run LDA
    print(f"Running LDA to find {N_TOPICS} topics...")
    lda = LatentDirichletAllocation(n_components=N_TOPICS, max_iter=10,
                                    learning_method='online', learning_offset=50.,
                                    random_state=42)
    lda.fit(dtm)

    # 5. Display the Topics
    topic_summaries = display_topics(lda, feature_names, N_TOP_WORDS_PER_TOPIC)

    print("\nCalculating topic distribution for each dream...")
    # Get Topic Distribution per Dream
    topic_distribution = lda.transform(dtm)
    topic_col_names = [f'Topic_{i+1}' for i in range(N_TOPICS)]
    topic_df = pd.DataFrame(topic_distribution, columns=topic_col_names, index=df_analysis.index)

    # Merge with emotion labels
    results_df = pd.concat([df_analysis[['Emotion_Label', 'Video_type']], topic_df], axis=1)

    # Calculate average topic probabilities per emotion
    print("\nAverage Topic Probabilities per Emotion")
    grouped_topics_emotion = results_df.groupby('Emotion_Label')[topic_col_names].mean()
    print(grouped_topics_emotion)

    # Visualize the average distributions
    print("\nGenerating heatmap of average topic probabilities per emotion...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(grouped_topics_emotion.T, annot=True, fmt=".2f", cmap="viridis") # Transpose for better view
    plt.title('Average Topic Probability per Dream Emotion')
    plt.xlabel('Dream Emotion')
    plt.ylabel('Topic')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(PLOT_SAVE_PATH): os.makedirs(PLOT_SAVE_PATH)
    plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME_TOPIC_EMOTION)
    try:
        plt.savefig(plot_filepath)
        plt.close()
        print(f"Successfully saved topic-emotion heatmap to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        plt.show()


if __name__ == '__main__':
    run_topic_modeling()