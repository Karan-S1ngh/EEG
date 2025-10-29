import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import warnings
from pandas.errors import SettingWithCopyWarning

FEATURES_FILE = 'dream_features.csv'
LOGBOOK_FILE = 'master_logbook.csv'
MODEL_SAVE_PATH = 'saved_models'
MODEL_NAME = 'multimodal_model_3class.pkl'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def train_multimodal_3class_model():
    # 1. Load BOTH datasets
    try:
        eeg_df = pd.read_csv(FEATURES_FILE)
        logbook_df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError as e:
        print(f"Error: Missing data file: {e}")
        return

    print(f"Loaded {eeg_df.shape[0]} EEG feature sets.")
    print(f"Loaded {logbook_df.shape[0]} logbook entries.")

    # 2. Combine the datasets
    logbook_clean = logbook_df.dropna(subset=['Dream_content', 'Dream_emotion'])
    logbook_clean = logbook_clean.reset_index().rename(columns={'index': 'dream_idx'})
    eeg_df = eeg_df.reset_index().rename(columns={'index': 'dream_idx'})

    # Select specific columns from logbook to avoid duplicate column names where possible
    df_merged = pd.merge(eeg_df, logbook_clean[['dream_idx', 'Dream_content', 'Dream_emotion']],
                         on='dream_idx', how='inner', suffixes=('', '_log')) # Add suffix to logbook's Dream_emotion

    print(f"Merged dataset shape: {df_merged.shape}")

    # The emotion label from the logbook is now 'Dream_emotion_log' due to the suffix
    emotion_column_name = 'Dream_emotion_log'

    # 3. Prepare data for 3-class classification
    # Use the correct emotion column name in dropna
    df_clean = df_merged.dropna(subset=['Dream_content', emotion_column_name])

    def map_emotion_3class(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion: return 'Negative'
        if 'happy' in emotion: return 'Positive'
        if 'calm' in emotion: return 'Calm'
        return None

    df_clean['Emotion_Label'] = df_clean[emotion_column_name].apply(map_emotion_3class)
    df_model = df_clean.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for 3-class multimodal modeling: {df_model.shape[0]} samples.")
    print("\n3-Class Class distribution")
    print(df_model['Emotion_Label'].value_counts())

    # 4. Define Features (X) and Target (y)
    eeg_feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    nlp_feature_column = 'Dream_content'
    X = df_model[eeg_feature_columns + [nlp_feature_column]]
    y = df_model['Emotion_Label']
    print(f"\nTraining model using {len(eeg_feature_columns)} EEG features + 1 NLP feature.")

    # 5. Create Preprocessing Pipeline
    numeric_transformer = StandardScaler()
    text_transformer = TfidfVectorizer(stop_words='english', max_features=1000)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, eeg_feature_columns),
            ('txt', text_transformer, nlp_feature_column)
        ],
        remainder='drop'
    )

    # 6. Create Full ML Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # 7. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Hyperparameter Tuning (GridSearchCV)
    print("\nStarting Hyperparameter Tuning (GridSearchCV) for 3 classes...")
    param_grid = {
        'preprocessor__txt__max_df': [0.75, 1.0],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 20],
        'classifier__min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    print("Tuning complete.")
    print(f"Found best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # 9. Evaluate the *best* model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation (3-Class Multimodal EEG+NLP) ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 33.3%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 10. Save the final *tuned* model
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath)
    print(f"\nSuccessfully saved 3-CLASS MULTIMODAL model to '{model_filepath}'")


if __name__ == '__main__':
    train_multimodal_3class_model()