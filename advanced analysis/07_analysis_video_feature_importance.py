import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

project_dir = os.getcwd() 
POWER_FEATURES_FILE = os.path.join(project_dir, 'dream_features.csv')
MODEL_SAVE_PATH = os.path.join(project_dir, 'saved_models')
MODEL_NAME = 'video_predictor_rf.pkl'

if project_dir not in sys.path: sys.path.append(project_dir)
warnings.filterwarnings('ignore')


def analyze_video_feature_importance():
    """
    Trains a model to predict Video_type from EEG features and extracts
    the importance of each feature in the classification task.
    """
    # 1. Load Power features
    try:
        df = pd.read_csv(POWER_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"Error: Missing feature file. Tried to access: {POWER_FEATURES_FILE}")
        return

    # 2. Prepare data
    df_analysis = df.dropna(subset=['Video_type']).copy()
    
    # Filter only for Positive, Negative, Neutral videos
    valid_videos = ['Positive', 'Negative', 'Neutral']
    df_model = df_analysis[df_analysis['Video_type'].isin(valid_videos)]
    
    if df_model.shape[0] < 50:
        print(f"Error: Insufficient data for video classification ({df_model.shape[0]} samples).")
        return

    print(f"Analyzing {df_model.shape[0]} dreams for video prediction.")

    # 3. Define Features (X) and Target (y)
    feature_columns = [col for col in df_model.columns if col.endswith('_power')]
    
    X = df_model[feature_columns]
    y = df_model['Video_type']

    # 4. Scale and Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train Model (Random Forest)
    print("\nTraining Random Forest to predict Video Type from EEG...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)

    # 6. Evaluate and Extract Feature Importance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nVideo Type Prediction Results")
    print(f"Accuracy: {accuracy * 100:.2f}% (Baseline guess is 33.3%)")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    # Extract Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Most Sensitive EEG Markers to Pre-Sleep Video")
    print(importance_df.head(10).to_markdown(index=False))

    # 7. Save Model
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    print(f"\nSuccessfully saved model to '{MODEL_NAME}'")


if __name__ == '__main__':
    analyze_video_feature_importance()