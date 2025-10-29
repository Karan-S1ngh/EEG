import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from pandas.errors import SettingWithCopyWarning

project_dir = os.getcwd() 
FEATURES_FILE = os.path.join(project_dir, 'entropy_features.csv')
MODEL_SAVE_PATH = os.path.join(project_dir, 'saved_models')

MODEL_NAME = 'entropy_model_3class.pkl'
SCALER_NAME = 'entropy_scaler_3class.pkl'

if project_dir not in sys.path:
     sys.path.append(project_dir)

print(f"DEBUG: Current Working Directory: {project_dir}")
print(f"DEBUG: Looking for features file at: {FEATURES_FILE}")
print(f"DEBUG: Saving models to: {MODEL_SAVE_PATH}")

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def train_entropy_3class_model():
    # 1. Load the entropy feature dataset
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Entropy features file not found: '{FEATURES_FILE}'")
        print("Please check the path. Ensure 'entropy_features.csv' is in the directory you are running this script from.")
        print("Also ensure build_alternative_features.py ran successfully.")
        return

    print(f"Loaded entropy dataset with {df.shape[0]} samples.")

    # 2. Prepare data for classification
    df_model = df.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for modeling: {df_model.shape[0]} samples.")
    print("\n3-Class Class distribution")
    print(df_model['Emotion_Label'].value_counts())

    # 3. Define Features (X) and Target (y)
    feature_columns = [col for col in df_model.columns if col.endswith('_PermutationEntropy')]
    if len(feature_columns) != 6:
        print(f"Warning: Expected 6 entropy features, found {len(feature_columns)}.")
        if len(feature_columns) == 0: return
    print(f"\nTraining model using {len(feature_columns)} entropy features...")

    X = df_model[feature_columns]
    y = df_model['Emotion_Label']

    # 4. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Hyperparameter Tuning (GridSearchCV)
    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    print("Tuning complete.")
    print(f"Found best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # 7. Evaluate the *best* model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation (3-Class Entropy RF)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 33.3%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save the final model and scaler
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath)
    scaler_filepath = os.path.join(MODEL_SAVE_PATH, SCALER_NAME)
    joblib.dump(scaler, scaler_filepath)
    print(f"\nSuccessfully saved 3-CLASS ENTROPY model to '{model_filepath}'")
    print(f"Successfully saved scaler to '{scaler_filepath}'")


if __name__ == '__main__':
    train_entropy_3class_model()