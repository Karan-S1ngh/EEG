import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

FEATURES_FILE = 'dream_features.csv' 
MODEL_SAVE_PATH = 'saved_models'   
MODEL_NAME = 'emotion_model_2class_calm.pkl'
SCALER_NAME = 'scaler_2class_calm.pkl'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_2class_model():
    """
    Trains a 2-class model (Calm vs. Not Calm) to get a higher accuracy.
    Uses the full 30-feature set and GridSearchCV for best performance.
    """
    # 1. Load the 30-feature dataset
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Features file not found: '{FEATURES_FILE}'")
        print("Please run build_features.py first.")
        return

    print(f"Loaded feature dataset with {df.shape[0]} samples.")

    # 2. Prepare data for 2-class classification
    df_clean = df.dropna(subset=['Dream_emotion'])

    def map_emotion_2class(emotion):
        emotion = str(emotion).lower()
        if 'calm' in emotion:
            return 'Calm'
        # Group "Positive" and "Negative" together
        if 'happy' in emotion or 'unhappy' in emotion or 'sad' in emotion:
            return 'Not Calm'
        return None

    # Apply the mapping safely
    df_clean = df_clean.copy() # Avoid SettingWithCopyWarning
    df_clean.loc[:, 'Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion_2class)

    df_model = df_clean.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for 2-class modeling: {df_model.shape[0]} samples.")
    print("\n2-Class Class distribution")
    print(df_model['Emotion_Label'].value_counts())
    print("(This is almost perfectly balanced!)")

    # 3. Define Features (X) and Target (y)
    feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    print(f"\nTraining model using {len(feature_columns)} features...")

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
    print("This may take a minute or two...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
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

    print(f"\nModel Evaluation (2-Class Calm Tuned RF)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("(Baseline guess is 50%)")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save the final *tuned* model and scaler
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(best_model, model_filepath)

    scaler_filepath = os.path.join(MODEL_SAVE_PATH, SCALER_NAME)
    joblib.dump(scaler, scaler_filepath)

    print(f"\nSuccessfully saved 2-CLASS CALM model to '{model_filepath}'")
    print(f"Successfully saved 2-CLASS CALM scaler to '{scaler_filepath}'")


if __name__ == '__main__':
    train_2class_model()