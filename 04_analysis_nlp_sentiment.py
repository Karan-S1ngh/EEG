import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
from pandas.errors import SettingWithCopyWarning

LOGBOOK_FILE = 'master_logbook.csv' # Adjusted path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def run_vader_analysis():
    """
    Uses the pre-trained VADER model to get a sentiment score
    for each dream report and compares it to the human-given label.
    """
    # 1. Load the master logbook
    try:
        df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError:
        print(f"Error: Logbook file not found: '{LOGBOOK_FILE}'")
        print("Please run data_loader.py first.")
        return

    print(f"Loaded master logbook with {df.shape[0]} samples.")

    # 2. Prepare data for classification
    df_clean = df.dropna(subset=['Dream_content', 'Dream_emotion'])

    # Use our bug-fixed 3-class mapping
    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion:
            return 'Negative'
        if 'happy' in emotion:
            return 'Positive'
        if 'calm' in emotion:
            return 'Neutral' # VADER uses 'Neutral', so let's match it
        return None

    df_clean['Human_Label'] = df_clean['Dream_emotion'].apply(map_emotion)
    df_model = df_clean.dropna(subset=['Human_Label'])

    print(f"Cleaned data for VADER analysis: {df_model.shape[0]} samples.")

    # 3. Initialize VADER and make predictions
    analyzer = SentimentIntensityAnalyzer()

    vader_predictions = []
    for text in df_model['Dream_content']:
        score = analyzer.polarity_scores(text)
        compound_score = score['compound']

        if compound_score >= 0.05:
            vader_predictions.append('Positive')
        elif compound_score <= -0.05:
            vader_predictions.append('Negative')
        else:
            vader_predictions.append('Neutral')

    df_model['VADER_Prediction'] = vader_predictions

    # 4. Evaluate the VADER model
    accuracy = accuracy_score(df_model['Human_Label'], df_model['VADER_Prediction'])

    print(f"\nVADER Sentiment Model Evaluation")
    print("Comparing pre-trained VADER model to human labels")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(df_model['Human_Label'], df_model['VADER_Prediction'], zero_division=0))

    print("\nThis accuracy shows how often the 'AI' sentiment model agreed with the human's reported emotion.")


if __name__ == '__main__':
    run_vader_analysis()