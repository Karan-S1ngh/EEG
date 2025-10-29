import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

FEATURES_FILE = 'dream_features.csv' 
MODEL_SAVE_PATH = 'saved_models'     
MODEL_NAME = 'emotion_model.pkl'
SCALER_NAME = 'scaler.pkl'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_emotion_model():
    """
    This is our best 3-class EEG model (V4).
    Trains the Random Forest model on the 30+ features and saves it.
    """
    # 1. Load the clean 30-feature dataset
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Features file not found: '{FEATURES_FILE}'")
        print("Please run build_features.py first.")
        return

    print(f"Loaded feature dataset with {df.shape[0]} samples.")

    # 2. Prepare data for classification (with the bug fix)
    df_clean = df.dropna(subset=['Dream_emotion'])

    def map_emotion(emotion):
        emotion = str(emotion).lower()
        if 'unhappy' in emotion or 'sad' in emotion:
            return 'Negative'
        if 'happy' in emotion:
            return 'Positive'
        if 'calm' in emotion:
            return 'Calm'
        return None

    df_clean['Emotion_Label'] = df_clean['Dream_emotion'].apply(map_emotion)
    df_model = df_clean.dropna(subset=['Emotion_Label'])

    print(f"Cleaned data for modeling: {df_model.shape[0]} samples.")
    print("\nClass distribution")
    print(df_model['Emotion_Label'].value_counts())

    # 3. Define Features (X) and Target (y)
    feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    print(f"\nTraining model using {len(feature_columns)} features")

    X = df_model[feature_columns]
    y = df_model['Emotion_Label']

    # 4. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Train the Random Forest (with class weighting)
    print("Training Random Forest Classifier (Our best 3-class model)")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    # We use the original y_train, not resampled
    model.fit(X_train, y_train)

    # 7. Evaluate the model
    print("Model training complete.")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Evaluation (Random Forest 3-Class)")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Save the final model and scaler
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    joblib.dump(model, model_filepath) # Save the RF model

    scaler_filepath = os.path.join(MODEL_SAVE_PATH, SCALER_NAME)
    joblib.dump(scaler, scaler_filepath) # Save the scaler

    print(f"\nSuccessfully saved BEST 3-CLASS model to '{model_filepath}'")
    print(f"Successfully saved scaler to '{scaler_filepath}'")


if __name__ == '__main__':
    train_emotion_model()