# EEG Dream Emotion Analysis Project

## 📖 Overview

This project is a comprehensive analysis of the **DEED (Dream Emotion Evaluation Dataset)**, exploring the complex relationships between brain activity (EEG), dream content (NLP), pre-sleep stimuli (videos), and subjective emotion.

The project involves several key stages:
1.  **Data Processing:** A full pipeline to load, clean, and merge raw metadata and EEG (`.mat`) files.
2.  **Advanced Feature Engineering:** Extraction of four distinct feature sets:
    * EEG Band Power (30 features)
    * EEG Entropy (6 features)
    * EEG Connectivity (75 features)
    * NLP Topic Modeling (10 features)
3.  **Machine Learning:** Training and evaluation of over a dozen models to find the most accurate predictors of dream emotion and video stimulus.
4.  **Statistical Analysis:** Deep-dive statistical analysis to find significant correlations between brain activity, dream themes, and external stimuli.
5.  **Interactive Deployment:** A multi-tab Streamlit dashboard that showcases all key findings, including live model predictions and an AI-powered dream interpreter.

---

## 📁 Project Structure

Root Folder
│
├── 📁 advanced analysis/                    # Advanced EEG–NLP–Video analysis scripts
│   ├── 01_connectivity_processor.py         # Test script for connectivity on a single dream
│   ├── 02_build_connectivity_features.py    # Builds -> connectivity_features.csv
│   ├── 03_analysis_connectivity.py          # Analyzes connectivity features -> boxplot
│   ├── 04_analysis_nlp_topic_modeling.py    # Runs LDA topic modeling -> topic_emotion_heatmap.png
│   ├── 05_save_topic_scores.py              # Saves LDA topic probabilities -> topic_scores.csv
│   ├── 06_analysis_topic_connectivity.py    # Correlates topics vs. brain connectivity
│   ├── 07_analysis_video_feature_importance.py # EEG→Video RF model (81.2%) + top features
│   ├── _08_analysis_video_themes.py         # Analyzes relation between video type & dream topics
│   ├── 09_visualize_channels.py             # Plots 6-channel electrode map
│   └── 10_visualize_connectivity_heatmap.py # 6x6 Alpha-band connectivity heatmap
│
├── 📁 eeg_mat_files/                        # Raw EEG recordings (.mat files)
│
├── 📁 saved_models/                         # Trained machine learning models & scalers
│   ├── emotion_model.pkl                    # EEG Power (3-class)
│   ├── emotion_model_2class_calm.pkl        # EEG Power (2-class, ~61%)
│   ├── entropy_model_2class_calm.pkl        # EEG Entropy (2-class, ~62%)
│   ├── entropy_model_3class.pkl             # EEG Entropy (3-class, ~46%)
│   ├── multimodal_entropy_model.pkl         # EEG Entropy + Text (~64%)
│   ├── multimodal_full_model.pkl            # EEG Power + Entropy + Text (~68%)
│   ├── multimodal_model.pkl                 # 🏆 BEST MODEL — EEG Power + Text (93.5%)
│   ├── multimodal_model_3class.pkl          # EEG Power + Text (3-class, ~62%)
│   ├── nlp_emotion_model.pkl                # NLP Text only (TF-IDF)
│   ├── video_predictor_rf.pkl               # EEG→Video RF model (81.2%)
│   ├── scaler.pkl / scaler_2class_calm.pkl  # Preprocessing scalers
│   └── entropy_scaler_*.pkl                 # Scalers for entropy models
│
├── 📁 saved_plots/                          # All generated plots & visualizations (.png)
│   ├── channel_locations.png                # 6-channel map visualization
│   ├── connectivity_heatmap_alpha.png       # Alpha-band 6×6 connectivity matrix
│   ├── content_correlation_heatmap.png      # Keyword vs. EEG Power correlation
│   ├── significant_connectivity_boxplot.png # Most significant connectivity difference
│   ├── topic_connectivity_correlation.png   # Topics vs. Connectivity heatmap
│   ├── topic_emotion_heatmap.png            # Topics vs. Emotion heatmap
│   └── video_impact_plot.png                # EEG Power vs. Video Type
│
├── 📜 .env                                  # Stores secret `GEMINI_API_KEY` for app integration
├── 📜 .gitignore                            # Specifies files ignored by version control (e.g., .env)
│
├── 📜 01_analysis_emotion.py                # EEG Power (3-class emotion model)
├── 📜 02_analysis_emotion_2class.py         # EEG Power (2-class emotion model)
├── 📜 03_analysis_nlp_emotion.py            # NLP-based (TF-IDF) emotion model
├── 📜 04_analysis_nlp_sentiment.py          # VADER-based sentiment test (~54%)
├── 📜 05_analysis_multimodal.py             # 🏆 BEST model — EEG Power + Text (93.5%)
├── 📜 06_analysis_multimodal_3class.py      # 3-class multimodal model (~62%)
├── 📜 07_analysis_content.py                # Keyword correlation analysis → `content_correlation_heatmap.png`
├── 📜 08_analysis_video.py                  # Video impact analysis → `video_impact_plot.png`
├── 📜 09_analysis_entropy_emotion_3class.py # EEG Entropy (3-class model, ~46%)
├── 📜 10_analysis_entropy_emotion_2class.py # EEG Entropy (2-class model, ~62%)
├── 📜 11_analysis_multimodal_entropy.py     # EEG Entropy + Text (2-class, ~64%)
├── 📜 12_analysis_multimodal_ful.py         # EEG Power + Entropy + Text (2-class, ~68%)
│
├── 📜 alterntive_features.py                # Builds entropy_features.csv (6 entropy features)
├── 📜 app.py                                # Streamlit dashboard (multi-tab visualization & predictions)
├── 📜 build_features.py                     # Builds dream_features.csv (30 EEG Power features)
├── 📜 data_loader.py                        # Merges 3 raw Excel files → master_logbook.csv
├── 📜 eeg_processor.py                      # Extracts EEG Power features from .mat files
│
├── 📜 Emotional_ratings_excel_files.xlsx    # Raw: Dream text + emotion ratings
├── 📜 Status_identification_of_each_stage_of_EEG.xlsx # Raw: Recording stage timings
├── 📜 Video_list.xlsx                       # Raw: Video type metadata (Pos/Neg/Neu)
│
├── 📜 connectivity_features.csv             # Clean: 75 EEG Connectivity features
├── 📜 dream_features.csv                    # Clean: 30 EEG Power features
├── 📜 entropy_features.csv                  # Clean: 6 EEG Entropy features
├── 📜 master_logbook.csv                    # Clean: Combined metadata
├── 📜 topic_scores.csv                      # Clean: 10 NLP Topic probabilities
│
└── 📜 README.md                             # This documentation file

---

## ⚙️ Setup

1.  **Place Data:**
    * Place the 3 raw `.xlsx` files in the root project folder.
    * Place all raw `.mat` files inside the `eeg_mat_files/` folder.
2.  **Create `.env` file:**
    * Create a file named `.env` in the root folder.
    * Add your Gemini API key: `GEMINI_API_KEY="YOUR_API_KEY_HERE"`
3.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy scipy mne mne-connectivity openpyxl joblib scikit-learn matplotlib seaborn tqdm python-dotenv google-generativeai antropy
    ```
4.  **Download NLTK Data** (for Topic Modeling):
    ```bash
    python -m nltk.downloader stopwords
    ```

---

## ▶️ How to Run

Execute scripts from your main project directory (`D:\Projects\EEG (Sem 7 MP)`) in this order:

1.  **Generate All Feature Datasets:**
    ```bash
    python data_loader.py
    python build_features.py
    python alterntive_features.py
    python "advanced analysis/02_build_connectivity_features.py"
    python "advanced analysis/05_save_topic_scores.py"
    ```
    *Output: All `.csv` feature files (master_logbook, dream_features, etc.)*

2.  **Generate All Models & Plots:**
    ```bash
    # Run your most important analyses to generate the saved .pkl and .png files
    python 05_analysis_multimodal.py
    python "advanced analysis/07_analysis_video_feature_importance.py"
    python 08_analysis_video.py
    python "advanced analysis/04_analysis_nlp_topic_modeling.py"
    python "advanced analysis/03_analysis_connectivity.py"
    python "advanced analysis/09_visualize_channels.py"
    python "advanced analysis/10_visualize_connectivity_heatmap.py"
    # ... etc. for any other plots you want in the app
    ```
    *Output: All models in `saved_models/` and plots in `saved_plots/`*

3.  **Launch the Streamlit Dashboard:**
    ```bash
    streamlit run app.py
    ```
    *This will open the interactive dashboard in your web browser.*

---

## ✨ Key Findings Summary

* **Best Model (93.5% Accuracy):** A multimodal model (`multimodal_model.pkl`) combining **EEG Power features + NLP text** was the most accurate at predicting dream emotion (Calm vs. Not Calm). This proves that brain data and dream text contain complementary information.
* **Video Impact (81.2% Accuracy):** A model (`video_predictor_rf.pkl`) could predict the *type* of video a person watched (Positive/Negative/Neutral) with **81.2% accuracy** using *only* the EEG data from their subsequent dream. This is a major finding, proving pre-sleep stimuli leave a strong neural signature.
* **Brain Connectivity:** Found **19 statistically significant** differences in brain connectivity (coherence) between emotion groups, primarily in the **Alpha and Theta bands** (e.g., `alpha_Ch5_Ch3` [C3-Pz]). This suggests emotions are linked to *how* brain regions synchronize.
* **NLP Topic Modeling:** Discovered 10 latent "themes" in the dream reports. Found that while emotion correlated with topic prevalence (e.g., Topic 7 higher in Negative dreams), the pre-sleep *video type* had **no significant impact on the dream's themes**.
* **Feature Comparison:** EEG Power features (~61%) and EEG Entropy (complexity) features (~62%) were similarly predictive on their own, but the Power features provided the best synergy when combined with NLP.