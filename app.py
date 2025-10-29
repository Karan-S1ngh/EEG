import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from dotenv import load_dotenv
import io # Required for reading uploaded file in memory

import scipy.io
import mne
try:
    import antropy as ant
except ImportError:
    st.error("Antropy library not found. pip install antropy")
    st.stop()

# Import Google AI SDK
try:
    import google.generativeai as genai
except ImportError:
    st.error("Google AI SDK not found. pip install google-generativeai")
    st.stop()

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
st.sidebar.caption(f"API Key Loaded: {'Yes' if API_KEY else 'NO - Check .env'}")

PROJECT_DIR = os.getcwd()
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'saved_models')
PLOT_SAVE_PATH = os.path.join(PROJECT_DIR, 'saved_plots')
SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'
NUM_CHANNELS = 6
POWER_BANDS = {
    'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13],
    'beta': [13, 30], 'gamma': [30, 45]
}

llm_model = None
LLM_MODEL_NAME = 'gemini-2.5-flash'
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
        print(f"INFO: SDK Configured with model '{LLM_MODEL_NAME}'.")
    except Exception as e:
        st.error(f"Error configuring SDK/model '{LLM_MODEL_NAME}': {e}")
else:
    st.sidebar.error("GEMINI_API_KEY missing. LLM features disabled.")

warnings.filterwarnings('ignore')

@st.cache_resource
def load_model(model_name):
    """Loads a saved model/pipeline file."""
    path = os.path.join(MODEL_SAVE_PATH, model_name)
    try:
        model = joblib.load(path); print(f"Loaded {model_name}"); return model
    except FileNotFoundError: st.sidebar.warning(f"'{model_name}' missing."); return None
    except Exception as e: st.sidebar.error(f"Error loading '{model_name}': {e}"); return None

# Load all the models to showcase
multimodal_model = load_model('multimodal_model.pkl') # 93.5% model
video_predictor_model = load_model('video_predictor_rf.pkl') # 81.2% model
# NOTE: We are missing the scaler for the video_predictor_model
# The interactive demo for it will not be possible without re-running
# 07_analysis_video_feature_importance.py to also save its scaler.

# Define paths to all plot files
video_impact_plot_path = os.path.join(PLOT_SAVE_PATH, 'video_impact_plot.png')
topic_emotion_heatmap_path = os.path.join(PLOT_SAVE_PATH, 'topic_emotion_heatmap.png')
channel_locations_path = os.path.join(PLOT_SAVE_PATH, 'channel_locations.png')
connectivity_chord_path = os.path.join(PLOT_SAVE_PATH, 'connectivity_network_chord_style.png')
connectivity_boxplot_path = os.path.join(PLOT_SAVE_PATH, 'significant_connectivity_boxplot.png')
content_heatmap_path = os.path.join(PLOT_SAVE_PATH, 'content_correlation_heatmap.png')

@st.cache_data(show_spinner=False)
def get_gemini_sdk_response_simple(prompt_text):
    if llm_model is None: return "Error: LLM Model not configured."
    try:
        response = llm_model.generate_content(prompt_text)
        if hasattr(response, 'text'):
             cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
             return cleaned_text
        else:
            try: block_reason = response.prompt_feedback.block_reason; return f"Error: Blocked ({block_reason})."
            except Exception: return "Error: Could not extract text."
    except Exception as e:
        st.error(f"Error calling Google AI SDK: {e}"); return f"Error: SDK call failed."


# LIVE EEG FEATURE EXTRACTION (Used by Tab 2)
@st.cache_data(show_spinner=False)
def _app_calculate_psd(eeg_segment, srate):
    features = {}
    try:
        num_channels = eeg_segment.shape[0]; ch_names = [f'EEG {i+1}' for i in range(num_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=['eeg']*num_channels)
        raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False); raw.set_montage('standard_1020', on_missing='ignore')
        raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45.0, n_fft=srate, verbose=False)
        psd, freqs = spectrum.get_data(return_freqs=True)
        for ch_index in range(num_channels):
            ch_name = f'ch{ch_index + 1}'
            for band_name, (fmin, fmax) in POWER_BANDS.items():
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_power = np.log10(np.mean(psd[ch_index, idx_band]))
                features[f'{ch_name}_{band_name}_power'] = band_power
    except Exception as e: st.error(f"PSD Error: {e}"); return None
    return features

def process_live_mat_file(file_bytes):
    try:
        mat_data = scipy.io.loadmat(io.BytesIO(file_bytes))
        eeg_segment = mat_data[EEG_VARIABLE_NAME]
        if eeg_segment.shape[0] != NUM_CHANNELS:
            st.error(f"Error: .mat file has {eeg_segment.shape[0]} channels, expected {NUM_CHANNELS}.")
            return None
        info = mne.create_info(ch_names=[f'Ch{i+1}' for i in range(NUM_CHANNELS)], sfreq=SAMPLING_RATE, ch_types=['eeg']*NUM_CHANNELS)
        raw = mne.io.RawArray(eeg_segment * 1e-6, info, verbose=False)
        raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
        filtered_eeg_segment = raw.get_data()
        power_features = _app_calculate_psd(filtered_eeg_segment, SAMPLING_RATE)
        return power_features # Return only the 30 power features
    except KeyError: st.error(f"Error: '{EEG_VARIABLE_NAME}' not found in .mat file."); return None
    except Exception as e: st.error(f"Error processing .mat file: {e}"); return None

# Helper Function for Multimodal Prediction
def predict_emotion_multimodal(eeg_power_features, dream_text):
    if multimodal_model is None: return "Error: Multimodal model missing.", None
    input_df = pd.DataFrame([eeg_power_features]); input_df['Dream_content'] = dream_text
    try:
        pred = multimodal_model.predict(input_df)
        prob = multimodal_model.predict_proba(input_df)
        if prob.shape == (1, len(multimodal_model.classes_)): return pred[0], prob[0]
        else: return pred[0], None
    except Exception as e: return f"Pred error: {e}", None


# Streamlit App Layout
st.set_page_config(layout="wide", page_title="Dream Analysis Dashboard")
st.title("🌙 Dream Analysis Dashboard ✨")
st.caption(f"Final Year Project by Karan | Analyzing the DEED Dataset")

# Sidebar Navigation 
st.sidebar.image("https://placehold.co/300x100/EEE/313131?text=Dream+Insights", use_container_width=True) # Fixed
st.sidebar.header("Dashboard Sections")
analysis_options = [
    "💬 Simple Dream Analyzer (AI)",   
    "🏆 Model 1: Emotion Prediction (93.5%)", 
    "🎬 Model 2: Video Prediction (81.2%)", 
    "📊 Research Plots & Visualizations"
]
if 'page' not in st.session_state: 
   st.session_state.page = analysis_options[0]
analysis_choice = st.sidebar.radio("Navigation:", analysis_options, index=analysis_options.index(st.session_state.page), key="nav_radio")
if analysis_choice != st.session_state.page: 
   st.session_state.page = analysis_choice
   st.rerun()
st.sidebar.info("This app showcases NLP analysis and advanced machine learning findings from EEG dream data.")


# Tab 1: Simple LLM Dream Analyzer (Default)
if st.session_state.page == "💬 Simple Dream Analyzer (AI)":
    st.header("💬 Simple Dream Analysis (AI-Powered)")
    st.markdown("Enter your dream description below. An AI (Gemini LLM) will analyze its **sentiment** and provide a **general interpretation** based on common symbols and themes.")
    if llm_model is None: 
        st.error("LLM model failed to initialize."); 
        st.stop()
    dream_text_input_llm = st.text_area("Enter your dream here:", height=200, placeholder="Describe your dream")
    if 'llm_sentiment' not in st.session_state: 
        st.session_state.llm_sentiment = ""
    if 'llm_interpretation' not in st.session_state: 
        st.session_state.llm_interpretation = ""
    analyze_button_llm = st.button("Analyze My Dream (AI)", type="primary")
    if analyze_button_llm:
        if not dream_text_input_llm:
            st.warning("Please enter description."); 
            st.session_state.llm_sentiment = ""; 
            st.session_state.llm_interpretation = ""
        else:
            st.session_state.llm_sentiment = "Analyzing"; 
            st.session_state.llm_interpretation = "Analyzing"
            with st.spinner("Analyzing sentiment..."):
                s_prompt = (f"Output only 'Positive', 'Negative', or 'Neutral'.\n\nAnalyze sentiment:\nDream: \"{dream_text_input_llm}\"\nSentiment:")
                s_response = get_gemini_sdk_response_simple(s_prompt)
                valid=["Positive","Negative","Neutral"]; found=None; clean="".join(c for c in s_response if c.isalnum())
                for s in valid:
                     if s.lower() == clean.lower(): 
                        found = s; 
                        break
                if found: 
                    st.session_state.llm_sentiment = found
                elif "Error" in s_response: 
                    st.session_state.llm_sentiment = s_response
                else: 
                    st.session_state.llm_sentiment = f"Unknown ({s_response[:50]})"
            with st.spinner("Generating interpretation"):
                i_prompt = (f"You are a gentle dream interpreter\n\nBriefly interpret (2-4 sentences):\nDream: \"{dream_text_input_llm}\"")
                st.session_state.llm_interpretation = get_gemini_sdk_response_simple(i_prompt)
    if st.session_state.llm_sentiment and st.session_state.llm_sentiment != "Analyzing":
        st.subheader("Sentiment Analysis (AI):");
        if "Error" in st.session_state.llm_sentiment or "Unknown" in st.session_state.llm_sentiment: 
            st.error(st.session_state.llm_sentiment)
        else: s_disp=st.session_state.llm_sentiment; 
        s_emo={"Positive":"😊","Negative":"😟","Neutral":"😐"}; 
        st.metric("Sentiment", f"{s_disp} {s_emo.get(s_disp,'')}")
    if st.session_state.llm_interpretation and st.session_state.llm_interpretation != "Analyzing":
        st.subheader("General Interpretation (AI):");
        if "Error" in st.session_state.llm_interpretation: 
            st.error(st.session_state.llm_interpretation)
        else: 
            st.info("💭 General interpretation:"); 
            st.write(st.session_state.llm_interpretation)


# Tab 2: Multimodal Emotion Prediction (93.5% Model)
elif st.session_state.page == "🏆 Model 1: Emotion Prediction (93.5%)":
    st.header("🏆 Research Model 1: Multimodal Emotion Prediction")
    st.markdown("This model predicts dream emotion **(Calm vs. Not Calm)** by combining **EEG Power Features** and **Dream Text (NLP)**.")
    st.success("📈 **Best Accuracy: 93.5%**")
    st.markdown("This demonstrates the power of multimodal analysis, as models using only EEG (~61%) or only text (~55%) were far less accurate.")

    if multimodal_model is None:
        st.error("Model file 'multimodal_model.pkl' not found. This demo is disabled.")
    else:
        st.subheader("Interactive Model Demo")
        st.info("Upload a `.mat` file from the dataset and paste its corresponding dream text to get a live prediction.")
        
        uploaded_mat_file = st.file_uploader("Upload EEG Segment (.mat file)", type=["mat"], key="multi_mat")
        dream_text_input = st.text_area("Enter Corresponding Dream Text:", "", key="multi_text")

        if st.button("🔮 Predict Emotion (Multimodal)", type="primary"):
            if uploaded_mat_file is not None and dream_text_input:
                with st.spinner("Processing .mat file & predicting..."):
                    file_bytes = uploaded_mat_file.getvalue()
                    power_features = process_live_mat_file(file_bytes)
                    
                    if power_features:
                        res = predict_emotion_multimodal(power_features, dream_text_input)
                        if isinstance(res,tuple) and len(res)==2:
                            pred, prob = res
                            if prob is not None: st.success(f"Predicted: **{pred}**");
                            try: 
                                classes=list(multimodal_model.classes_); 
                                idx=classes.index(pred); 
                                conf=prob[idx]; 
                                st.progress(conf); 
                                st.metric("Conf.", f"{conf*100:.1f}%")
                            except Exception as e: 
                                st.warning(f"Conf err: {e}")
                            else: 
                                st.success(f"Predicted: **{pred}** (Conf failed)")
                        else: 
                            st.error(f"Pred Error: {res}")
            else:
                st.warning("Please upload a .mat file AND enter dream text.")


# Tab 3: Video Prediction Model (81.2% Model)
elif st.session_state.page == "🎬 Model 2: Video Prediction (81.2%)":
    st.header("🎬 Research Model 2: Video Type Prediction from EEG")
    st.markdown("This model predicts the *type of video* (Positive, Negative, or Neutral) the person watched *before sleep* using **only the 30 EEG power features** from their dream.")
    st.success("📈 **Finding:** The model achieved **81.2% accuracy** (vs. 33.3% chance). This is a strong result, proving that pre-sleep emotional stimuli leave a detectable signature in the brain's activity during subsequent dreams.")
    
    if video_predictor_model is None:
        st.error("Model file 'video_predictor_rf.pkl' not found. This feature is disabled.")
    else:
        # We don't have the scaler, so we can't build a live demo.
        # Instead, we display the model's key finding: Feature Importances.
        st.subheader("Most Sensitive EEG Markers (Feature Importance)")
        st.markdown("This table shows the Top 10 EEG features the model used to distinguish between video types. This tells us *which* brain signals were most affected by the videos.")
        try:
            # Re-create power_cols based on how video_predictor_rf.pkl was trained
            power_cols = [f'ch{i+1}_{b}_power' for i in range(6) for b in ['delta','theta','alpha','beta','gamma']]
            imp_df = pd.DataFrame({'Feature': power_cols, 'Importance': video_predictor_model.feature_importances_}).sort_values('Importance', ascending=False)
            st.dataframe(imp_df.head(10), use_container_width=True)
            st.caption("Features like Gamma and Theta power appear to be highly sensitive to the pre-sleep stimulus.")
        except Exception as e:
            st.warning(f"Could not display feature importances: {e}")
            st.write("Top features (from script output) included: `ch6_gamma_power`, `ch4_gamma_power`, `ch5_theta_power`")


# Tab 4: All Plots & Visualizations
elif st.session_state.page == "📊 Research: Plots & Visualizations":
    st.header("📊 Research Plots & Visualizations")
    st.markdown("This tab displays all the key visual findings from the various research analyses.")
    
    # Channel Locations 
    st.subheader("Assumed EEG Channel Locations")
    if os.path.exists(channel_locations_path):
        st.image(channel_locations_path, width=400)
        st.caption(f"Plot from `visualize_channels.py`: Assumed 6-channel EEG layout (Fz, Cz, Pz, Oz, C3, C4) based on the 10-20 system, providing spatial context for the EEG features.")
    else:
        st.warning(f"Plot missing: '{os.path.basename(channel_locations_path)}'.")
    st.markdown("---")
    
    # Video Impact
    st.subheader("Plot: Video Impact on EEG Power")
    if os.path.exists(video_impact_plot_path):
        st.image(video_impact_plot_path)
        st.caption(f"Plot from `analysis_video.py`: This bar chart visualizes the statistical findings, showing average EEG power differences based on the video watched.")
    else:
        st.warning(f"Plot missing: '{os.path.basename(video_impact_plot_path)}'.")
    st.markdown("---")

    # Topic Modeling
    st.subheader("Plot: Dream Topics vs. Emotion")
    if os.path.exists(topic_emotion_heatmap_path):
        st.image(topic_emotion_heatmap_path)
        st.caption(f"Plot from `analysis_nlp_topic_modeling.py`: This heatmap shows the average prevalence of the 10 discovered dream topics (e.g., 'School', 'Family', 'Conflict') for each reported emotion.")
    else:
        st.warning(f"Heatmap missing: '{os.path.basename(topic_emotion_heatmap_path)}'.")
    st.markdown("---")
    
    # Connectivity
    st.subheader("Plots: Brain Connectivity & Emotion")
    st.markdown("These plots visualize findings from the advanced connectivity analysis.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Significant Connection (Boxplot)**")
        if os.path.exists(connectivity_boxplot_path):
            st.image(connectivity_boxplot_path)
            st.caption(f"Plot from `analysis_connectivity.py`: Distribution of the *most* significant connection (`alpha_Ch5_Ch3`) across emotions.")
        else:
            st.warning(f"Boxplot missing: '{os.path.basename(connectivity_boxplot_path)}'.")
    with col2:
        st.markdown("**Connectivity Network (Chord Diagram)**")
        if os.path.exists(connectivity_chord_path):
            st.image(connectivity_chord_path)
            st.caption(f"Plot from `visualize_connectivity_network.py`: Avg. Alpha Band Connectivity for *Significant* Connections.")
        else:
            st.warning(f"Chord plot missing: '{os.path.basename(connectivity_chord_path)}'.")
    st.markdown("---")

    # Content Correlation
    st.subheader("Plot: Keyword vs. EEG Power Correlation")
    if os.path.exists(content_heatmap_path):
        st.image(content_heatmap_path)
        st.caption(f"Plot from `analysis_content.py`: Correlation between Top 20 Dream Keywords (TF-IDF) and 30 EEG Features (Log Power).")
    else:
        st.warning(f"Heatmap missing: '{os.path.basename(content_heatmap_path)}'.")
    st.caption("This heatmap visualizes the direct correlation between the importance of specific *words* and EEG power. The correlations were found to be generally weak.")

# Footer
st.markdown("---")