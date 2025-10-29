import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

LOGBOOK_FILE = 'master_logbook.csv' # Adjusted path
MODEL_SAVE_PATH = 'saved_models'    # Adjusted path
MODEL_NAME = 'nlp_emotion_model.pkl'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_nlp_model():
    """
    Trains a model predicting Dream_emotion from Dream_content text
    using TF-IDF and RandomForest.
    """
    # 1. Load the master logbook
    try:
        # Need encoding for this file
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
            return 'Calm'
        return None

    # Apply the mapping safely
    df_clean = df_clean.copy() # Avoid SettingWithCopyWarning
    df_clean.loc[:, 'Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion)
    df_model = df_clean.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for NLP modeling: {df_model.shape[0]} samples.")
    print("\nClass distribution")
    print(df_model['Emotion_Label'].value_counts())

    # 3. Define Features (X) and Target (y)
    X = df_model['Dream_content']
    y = df_model['Emotion_Label']

    # 4. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Create an NLP Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # 6. Hyperparameter Tuning (GridSearchCV)
    print("\nStarting Hyperparameter Tuning (GridSearchCV)")

    param_grid = {
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__min_df': [1, 5],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 20]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    print("Tuning complete.")
    print(f"Found best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    # 7. Evaluate the *best* model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Evaluation (NLP Text Model)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 33.3%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save the final *tuned* model (the whole pipeline)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath) # Save the entire pipeline

    print(f"\nSuccessfully saved NLP model to '{model_filepath}'")


if __name__ == '__main__':
    train_nlp_model()