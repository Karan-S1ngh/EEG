import os
import sys
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
import warnings
from pandas.errors import SettingWithCopyWarning

project_dir = os.getcwd()
POWER_FEATURES_FILE = os.path.join(project_dir, 'dream_features.csv')
ENTROPY_FEATURES_FILE = os.path.join(project_dir, 'entropy_features.csv')
LOGBOOK_FILE = os.path.join(project_dir, 'master_logbook.csv')
MODEL_SAVE_PATH = os.path.join(project_dir, 'saved_models')
MODEL_NAME = 'multimodal_full_model.pkl'

if project_dir not in sys.path:
     sys.path.append(project_dir)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def train_multimodal_full_model():
    # 1. Load ALL THREE datasets
    try:
        power_df = pd.read_csv(POWER_FEATURES_FILE)
        entropy_df = pd.read_csv(ENTROPY_FEATURES_FILE)
        logbook_df = pd.read_csv(LOGBOOK_FILE, encoding='latin1', engine='python')
    except FileNotFoundError as e:
        print(f"Error: Missing data file: {e}")
        return

    print(f"Loaded {power_df.shape[0]} EEG power feature sets.")
    print(f"Loaded {entropy_df.shape[0]} EEG entropy feature sets.")
    print(f"Loaded {logbook_df.shape[0]} logbook entries.")

    # 2. Combine the datasets
    logbook_clean = logbook_df.dropna(subset=['Dream_content', 'Dream_emotion'])
    logbook_clean = logbook_clean.reset_index().rename(columns={'index': 'dream_idx'})

    power_df = power_df.reset_index().rename(columns={'index': 'dream_idx'})
    entropy_df = entropy_df.reset_index().rename(columns={'index': 'dream_idx'})

    # Merge logbook content/emotion onto power features
    df_merged_power = pd.merge(power_df, logbook_clean[['dream_idx', 'Dream_content', 'Dream_emotion']],
                         on='dream_idx', how='inner', suffixes=('_pow', '_log1')) # Add suffixes

    # Merge logbook content/emotion onto entropy features
    df_merged_entropy = pd.merge(entropy_df, logbook_clean[['dream_idx', 'Dream_content', 'Dream_emotion']],
                         on='dream_idx', how='inner', suffixes=('_ent', '_log2')) # Add suffixes

    # Now merge the two EEG datasets based on dream_idx
    entropy_cols_to_keep = ['dream_idx'] + [col for col in df_merged_entropy.columns if col.endswith('_PermutationEntropy')]
    # When merging, specify suffixes again to handle Online_id, Video_type, Emotion_Label if present in both
    df_full_merged = pd.merge(df_merged_power, df_merged_entropy[entropy_cols_to_keep],
                              on='dream_idx', how='inner', suffixes=('_from_power', '_from_entropy'))

    print(f"Fully merged dataset shape: {df_full_merged.shape}")
    print("\nColumns in final merged DataFrame:")
    print(list(df_full_merged.columns))

    # 3. Prepare data for 2-class classification
    df_clean = df_full_merged.dropna(subset=['Dream_content', 'Dream_emotion_log1']) # GUESSING the name for now

    def map_emotion_2class(emotion):
        emotion = str(emotion).lower()
        if 'calm' in emotion: return 'Calm'
        if 'happy' in emotion or 'unhappy' in emotion or 'sad' in emotion: return 'Not Calm'
        return None

    df_clean['Emotion_Label'] = df_clean['Dream_emotion_log1'].apply(map_emotion_2class) # GUESSING the name
    df_model = df_clean.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for 2-class full multimodal modeling: {df_model.shape[0]} samples.")
    print("\n2-Class Class distribution:")
    print(df_model['Emotion_Label'].value_counts())

    # 4. Define Features (X) and Target (y)
    power_feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    entropy_feature_columns = [col for col in df_model.columns if col.endswith('_PermutationEntropy')]
    nlp_feature_column = 'Dream_content' # Make sure this name is correct too
    if len(power_feature_columns) != 30 or len(entropy_feature_columns) != 6:
        print(f"Error: Incorrect number of EEG features found (Power={len(power_feature_columns)}, Entropy={len(entropy_feature_columns)}).")
        return

    X = df_model[power_feature_columns + entropy_feature_columns + [nlp_feature_column]]
    y = df_model['Emotion_Label']
    print(f"\nTraining model using {len(power_feature_columns)} Power + {len(entropy_feature_columns)} Entropy + 1 NLP feature.")

    # 5. Create Preprocessing Pipeline
    numeric_features = power_feature_columns + entropy_feature_columns
    numeric_transformer = StandardScaler()
    text_transformer = TfidfVectorizer(stop_words='english', max_features=1000)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('txt', text_transformer, nlp_feature_column)
        ],
        remainder='drop'
    )

    # 6. Create Full ML Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Hyperparameter Tuning
    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    param_grid = {
        'preprocessor__txt__max_df': [0.75, 1.0],
        'classifier__n_estimators': [100, 200], # Reduced estimators for speed
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

    # 9. Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation (Multimodal Full - Power+Entropy+NLP)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 50%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 10. Save
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath)
    print(f"\nSuccessfully saved FULL MULTIMODAL model to '{model_filepath}'")


if __name__ == '__main__':
    train_multimodal_full_model()