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

FEATURES_FILE = 'dream_features.csv'
LOGBOOK_FILE = 'master_logbook.csv'
MODEL_SAVE_PATH = 'saved_models'

MODEL_NAME = 'multimodal_model.pkl'
PREPROCESSOR_NAME = 'multimodal_preprocessor.pkl'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_multimodal_model():
    """
    Trains our most advanced model (V10).
    Combines 30 EEG features AND NLP text features to predict
    "Calm" vs. "Not Calm" for the highest possible accuracy.
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

    # 2. Combine the datasets
    
    # Let's clean the logbook for merging
    logbook_clean = logbook_df.dropna(subset=['Dream_content', 'Dream_emotion'])
    
    # We need a unique ID for each dream. Let's reset the index.
    eeg_df = eeg_df.reset_index()
    logbook_clean = logbook_clean.reset_index()
    
    if len(eeg_df) != len(logbook_clean[logbook_clean['Online_id'].isin(eeg_df['Online_id'])]):
        print("Warning: Merging dataframes. This might be imperfect.")
        df_merged = pd.merge(eeg_df, logbook_clean[['Online_id', 'Dream_content', 'Dream_emotion']], 
                             on=['Online_id', 'Dream_emotion'], how='left')
        df_merged = df_merged.drop_duplicates(subset=eeg_df.columns)
    else:
        print("Joining EEG features with dream text...")
        df_merged = eeg_df.join(logbook_clean['Dream_content'])

    # 3. Prepare data for 2-class classification
    df_clean = df_merged.dropna(subset=['Dream_content', 'Dream_emotion'])
    
    def map_emotion_2class(emotion):
        emotion = str(emotion).lower()
        if 'calm' in emotion:
            return 'Calm'
        if 'happy' in emotion or 'unhappy' in emotion or 'sad' in emotion:
            return 'Not Calm'
        return None 
        
    df_clean['Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion_2class)
    df_model = df_clean.dropna(subset=['Emotion_Label'])
    
    print(f"Cleaned data for 2-class multimodal modeling: {df_model.shape[0]} samples.")
    print("\n2-Class Class distribution")
    print(df_model['Emotion_Label'].value_counts())

    # 4. Define Features (X) and Target (y)
    eeg_feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    nlp_feature_column = 'Dream_content'
    
    X = df_model[eeg_feature_columns + [nlp_feature_column]]
    y = df_model['Emotion_Label']
    
    print(f"\nTraining model using {len(eeg_feature_columns)} EEG features + 1 NLP feature.")

    # 5. Create a Preprocessing Pipeline

    # Pipeline for numeric (EEG) features
    numeric_transformer = StandardScaler()
    
    # Pipeline for text (NLP) features
    text_transformer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Combine these preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, eeg_feature_columns),
            ('txt', text_transformer, nlp_feature_column)
        ],
        remainder='drop'
    )

    # 6. Create the Full ML Pipeline
    # This pipeline will:
    # 1. Preprocess (scale numeric, vectorize text)
    # 2. Train the classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Hyperparameter Tuning (GridSearchCV)
    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    print("This may take a minute or two...")
    
    # Tune parameters for both preprocessor and classifier
    param_grid = {
        'preprocessor__txt__max_df': [0.75, 1.0],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 20],
    }
    
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5, 
        n_jobs=-1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Tuning complete.")
    print(f"Found best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_

    # 9. Evaluate the *best* model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Evaluation (Multimodal EEG+NLP)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 50%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 10. Save the final *tuned* model and preprocessor
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
        
    # We save the *entire* best_model pipeline
    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath)
    
    print(f"\nSuccessfully saved MULTIMODAL model to '{model_filepath}'")


if __name__ == '__main__':
    train_multimodal_model()