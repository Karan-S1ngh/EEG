import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from dotenv import load_dotenv
import io

try:
    import scipy.io
    import mne
    import antropy as ant
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Missing library: {e}. Please install: `pip install scipy mne antropy google-generativeai`")
    st.stop()

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY", "")

PROJECT_DIR = os.getcwd()
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'saved_models')
PLOT_SAVE_PATH = os.path.join(PROJECT_DIR, 'saved_plots')
SAMPLING_RATE = 200
EEG_VARIABLE_NAME = 'Data'
NUM_CHANNELS = 6
POWER_BANDS = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 45]}
EEG_POWER_COLS = [f'ch{i+1}_{b}_power' for i in range(NUM_CHANNELS) for b in POWER_BANDS.keys()]

warnings.filterwarnings('ignore')

llm_model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        llm_model = genai.GenerativeModel('models/gemini-2.5-flash')
    except Exception as e: st.sidebar.error(f"LLM Error: {e}")

@st.cache_resource
def load_pkl(name):
    try: return joblib.load(os.path.join(MODEL_SAVE_PATH, name))
    except: return None

# Load ALL models for Tab 2 & 3
multimodal_model = load_pkl('multimodal_model.pkl')          # 93.5% (EEG+NLP)
eeg_model = load_pkl('emotion_model_2class_calm.pkl')        # ~61% (EEG Power)
nlp_model = load_pkl('nlp_emotion_model.pkl')                # ~60% (NLP Only)
video_model = load_pkl('video_predictor_rf.pkl')             # 81.2% (Video)
eeg_scaler = load_pkl('scaler_2class_calm.pkl')              # Scaler for EEG/Video


@st.cache_data(show_spinner=False)
def get_llm_response(prompt):
    if not llm_model: return "Error: LLM not configured."
    try:
        res = llm_model.generate_content(prompt)
        return res.text.strip().replace("```json", "").replace("```", "") if hasattr(res, 'text') else "Error: No text returned."
    except Exception as e: return f"Error: {e}"

def extract_eeg_features_live(file_bytes):
    """Extracts 30 Power + 6 Entropy features from a .mat file (NO CACHE)."""
    try:
        mat = scipy.io.loadmat(io.BytesIO(file_bytes))
        data = mat.get(EEG_VARIABLE_NAME)
        if data is None: 
            st.error(f"EEG data variable '{EEG_VARIABLE_NAME}' not found in .mat file.")
            return None
        
        if data.ndim == 1:
            st.error("EEG data is 1D. Expected 2D (channels x samples).")
            return None
        elif data.ndim > 2:
            st.warning(f"EEG data has {data.ndim} dimensions. Attempting to reshape to 2D.")
            data = data.reshape(data.shape[-2], data.shape[-1])
            
        if data.shape[0] != NUM_CHANNELS and data.shape[1] == NUM_CHANNELS:
            data = data.T # Transpose if (samples, channels)
        elif data.shape[0] != NUM_CHANNELS and data.shape[1] != NUM_CHANNELS:
            st.error(f"EEG data has {data.shape[0]}x{data.shape[1]} dimensions, expected {NUM_CHANNELS} channels. Cannot auto-transpose.")
            return None

        if data.shape[0] != NUM_CHANNELS:
            st.error(f"Processed EEG data has {data.shape[0]} channels, expected {NUM_CHANNELS}. Please verify .mat file structure.")
            return None
            
        info = mne.create_info([f'Ch{i+1}' for i in range(NUM_CHANNELS)], SAMPLING_RATE, ['eeg']*NUM_CHANNELS)
        raw = mne.io.RawArray(data * 1e-6, info, verbose=False) # MNE expects Volts
        raw.set_montage('standard_1020', on_missing='ignore')
        raw.filter(0.5, 45.0, verbose=False)
        
        # Power
        psd, freqs = raw.compute_psd(fmin=0.5, fmax=45.0, verbose=False).get_data(return_freqs=True)
        feats = {f'ch{i+1}_{b}_power': np.log10(np.mean(psd[i, (freqs>=fmin)&(freqs<=fmax)])) 
                 for i in range(NUM_CHANNELS) for b, (fmin, fmax) in POWER_BANDS.items()}
             
        return feats
    except Exception as e: 
        st.error(f"Error extracting EEG features: {e}")
        return None

def predict_emotion(features, text, model_type='multimodal'):
    """Unified prediction function for all emotion models."""
    try:
        if model_type == 'multimodal' and multimodal_model:
            df = pd.DataFrame([features])[EEG_POWER_COLS] # Ensure correct column order
            df['Dream_content'] = text
            model = multimodal_model
        elif model_type == 'eeg' and eeg_model and eeg_scaler:
            df = pd.DataFrame([features])[EEG_POWER_COLS]
            df = eeg_scaler.transform(df) # Scale for EEG-only
            model = eeg_model
        elif model_type == 'nlp' and nlp_model:
            df = pd.DataFrame({'Dream_content': [text]})
            model = nlp_model
        else: return "Error: Model missing", None

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        return pred, prob
    except Exception as e: return f"Error: {e}", None

def predict_video(features):
    """Predicts video type from EEG."""
    if not video_model or not eeg_scaler: return "Error: Model missing", None
    try:
        df = pd.DataFrame([features])[EEG_POWER_COLS]
        df_scaled = eeg_scaler.transform(df)
        return video_model.predict(df_scaled)[0], video_model.predict_proba(df_scaled)[0]
    except Exception as e: return f"Error: {e}", None

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Dream Analysis Dashboard", layout="wide")
st.sidebar.title("🌙 Dream Insights")
st.sidebar.info("Analysis of the DEED Dataset: EEG, NLP, and Emotion.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 AI Dream Analyzer", 
    "🧠 Emotion Prediction Research", 
    "🎬 Video Impact Research", 
    "🔬 Raw EEG Feature Extractor",
    "📊 All Visualizations",
    "🚀 Future & About"
])

# TAB 1: AI Analyzer
with tab1:
    st.header("Simple Dream Analysis (AI-Powered)")
    user_text = st.text_area("Describe your dream:", height=150, key="llm_dream_text")
    if st.button("✨ Analyze Dream (AI)", key="analyze_llm_button"):
        if user_text:
            with st.spinner("Thinking..."):
                sent = get_llm_response(f"Classify sentiment (Positive/Negative/Neutral) ONLY:\n{user_text}")
                interp = get_llm_response(f"Briefly interpret common themes (2-4 sentences) and symbolic elements. Keep it concise.:\n{user_text}")
            st.subheader("Results")
            st.metric("Sentiment", sent)
            st.info(f"**Interpretation:** {interp}")
        else: st.warning("Please enter your dream description to analyze.")

# TAB 2: Emotion Research
with tab2:
    st.header("🧠 Research: Predicting Dream Emotion (Calm vs. Not Calm)")
    st.write("Comparing different data modalities for emotion prediction.")
    
    col_a, col_b = st.columns(2)
    uploaded_file = st.file_uploader("Upload EEG (.mat) Segment", type=["mat"], key="emo_mat")
    
    # Container for shared EEG features
    if 'eeg_feats' not in st.session_state: st.session_state.eeg_feats = None
    
    if uploaded_file:
        if st.button("🔄 Process EEG File", key="process_eeg_button"):
            with st.spinner("Extracting features..."):
                st.session_state.eeg_feats = extract_eeg_features_live(uploaded_file.getvalue())
                if st.session_state.eeg_feats: st.success("EEG Features Extracted & Ready!")
                else: st.error("EEG extraction failed. Ensure it's a 6-channel .mat file.")

    st.markdown("---")
    
    st.subheader("Multimodal Model")
    st.caption("Accuracy: **93.5%**")
    if st.button("🔮 Predict (Multimodal)", key="predict_multimodal"):
        if st.session_state.eeg_feats:
            pred, prob = predict_emotion(st.session_state.eeg_feats, 'multimodal')
            if prob is not None: 
                st.metric("Predicted Emotion", f"{pred}", f"Confidence: {max(prob)*100:.1f}%")
                # st.bar_chart(pd.DataFrame({'Class': multimodal_model.classes_, 'Probability': prob}).set_index('Class'))
            else: st.error(pred)
        else: st.warning("Process the EEG file first.")

# TAB 3: Video Research
with tab3:
    st.header("🎬 Research: Pre-Sleep Video Type Prediction")
    st.write("This model predicts the emotional valence of a video watched before sleep (Positive, Negative, or Neutral) using subsequent dream EEG.")
    st.success("Accuracy: ~81.2%")
    vid_file = st.file_uploader("Upload EEG Segment (.mat) for Video Prediction", type=['mat'], key="vid_mat")
    if vid_file:
        if st.button("▶️ Predict Pre-Sleep Video Type", key="predict_video_button"):
            with st.spinner("Analyzing EEG..."):
                feats = extract_eeg_features_live(vid_file.getvalue())
                if feats:
                    pred, prob = predict_video(feats)
                    if prob is not None: 
                        st.metric("Predicted Video Type", f"{pred}", f"Confidence: {max(prob)*100:.1f}%")
                        # st.bar_chart(pd.DataFrame({'Class': video_model.classes_, 'Probability': prob}).set_index('Class'))
                    else: st.error(pred)
                else: st.error("EEG extraction failed. Ensure it's a 6-channel .mat file.")

with tab4:
    st.header("🔬 Raw EEG Power Feature Extractor")
    st.write("Upload an EEG (.mat) file to see the extracted power features across different frequency bands (Delta, Theta, Alpha, Beta, Gamma) for each of the 6 channels.")
    st.info(f"Expected EEG data variable name in .mat file: `{EEG_VARIABLE_NAME}`. Expected {NUM_CHANNELS} channels.")
    
    uploaded_raw_eeg_file = st.file_uploader("Upload Raw EEG Segment (.mat file)", type=["mat"], key="raw_eeg_extractor")
    
    if uploaded_raw_eeg_file:
        if st.button("Extract & Display Features", key="extract_raw_features_button"):
            with st.spinner("Extracting power features..."):
                raw_eeg_feats = extract_eeg_features_live(uploaded_raw_eeg_file.getvalue())
                
                if raw_eeg_feats:
                    st.subheader("Extracted EEG Power Features (log10 µV²):")
                    feats_df = pd.DataFrame([raw_eeg_feats]).T
                    feats_df.columns = ["Power (log10)"]
                    st.dataframe(feats_df)
                    
                    st.subheader("Visualization of Channel Power per Band")
                    # Create a more structured DataFrame for plotting
                    plot_data = []
                    for col, val in raw_eeg_feats.items():
                        parts = col.split('_')
                        channel = parts[0]
                        band = parts[1]
                        plot_data.append({'Channel': channel, 'Band': band, 'Power': val})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=plot_df, x='Channel', y='Power', hue='Band', palette='viridis', ax=ax)
                    ax.set_title('EEG Power Distribution Across Channels and Bands')
                    ax.set_ylabel('Log10 Power (µV²)')
                    ax.set_xlabel('EEG Channel')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    st.success("Features extracted and displayed!")
                else:
                    st.error("Failed to extract EEG features. Please check the .mat file format and content.")

# TAB 5: Visualizations
with tab5:
    st.header("📊 Research Visualizations")
    
    # Define plots with explanations
    plots_info = [
        {"title": "Average EEG Band Power by Pre-Sleep Video Type", 
         "filename": "video_impact_plot.png", 
         "description": "This bar chart displays the average logarithmic power of different EEG frequency bands (delta, theta, alpha, beta, gamma) during dreams, segmented by the emotional valence (Negative, Neutral, Positive) of the video watched prior to sleep. It helps visualize how pre-sleep emotional stimuli may influence subsequent brain activity patterns during dreaming."},
        {"title": "Dream Topics-Emotion Heatmap", 
         "filename": "topic_emotion_heatmap.png", 
         "description": "This heatmap visualizes the correlation between identified dream topics (e.g., 'Flight', 'Social') and reported dream emotions (e.g., Calm, Anxious). Darker colors indicate stronger associations."},
        {"title": "Significant Brain Connectivity Boxplots by Emotion", 
         "filename": "significant_connectivity_boxplot.png",
         "description": "Boxplots comparing the strength of specific functional EEG connections across different dream emotion categories. This highlights which neural pathways might be more active or coherent during 'Calm' versus 'Not Calm' dreams."},
        {"title": "EEG Functional Connectivity Network (Alpha Band Heatmap)", 
         "filename": "connectivity_heatmap_alpha.png",
         "description": "This heatmap specifically focuses on functional connectivity within the Alpha EEG band and its variations across different dream emotions. It might reveal altered states of consciousness related to dream content."},
        {"title": "EEG Electrode Placement for Emotion Recognition", 
         "filename": "combined_electrode_locations.png",
         "description": "To verify the importance of EEG from the frontal, front-temporal and temporal cortex for emotion recognition, electrode locations were mapped as shown in the figure. These selected regions are crucial for capturing brain activity relevant to emotional processing during dreaming."},
        {"title": "Dream Content Feature Correlation Heatmap", 
         "filename": "content_correlation_heatmap.png",
         "description": "A heatmap showing correlations between various NLP-extracted features from dream content (e.g., word frequencies, sentiment scores). This helps identify co-occurring themes or linguistic patterns."},
        {"title": "Dream Topic-Connectivity Correlation",
         "filename": "topic_connectivity_correlation.png",
         "description": "This visualization explores the relationship between specific dream topics (e.g., 'Aggression', 'Nature') and patterns of EEG functional connectivity, suggesting how neural network states relate to dream narrative content."},
        ]
    
    for plot_data in plots_info:
        title = plot_data["title"]
        filename = plot_data["filename"]
        description = plot_data["description"]
        path = os.path.join(PLOT_SAVE_PATH, filename)
        
        st.subheader(title)
        st.write(description)
        
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(f"Plot file not found: `{filename}` in `{PLOT_SAVE_PATH}`. Please ensure it exists.")
        st.markdown("---")

# TAB 6: Future Directions & About
with tab6:
    st.header("🚀 Future Directions & About This Project")
    st.markdown("---")

    st.subheader("Future Research & Product Ideas")
    st.write("""
        This dashboard represents a foundational step in automated dream analysis. Here are some exciting avenues for future development:
    """)
    st.markdown("""
    * **Real-time Dream Analysis:** Developing systems for live EEG monitoring during sleep to identify dream onset or emotional shifts in real-time.
    * **Personalized Interpretation:** Integrating user feedback to fine-tune AI interpretations for individual dreamers, recognizing personal symbols and experiences.
    * **Integration with Wearable Sleep Trackers:** Connecting with commercial sleep tracking devices (e.g., smart rings, headbands) to provide more accessible dream insights.
    * **Enhanced Emotion Classification:** Expanding beyond Calm/Not Calm to a broader spectrum of emotions (e.g., joy, fear, anxiety, wonder, confusion).
    * **Dream Recall Enhancement:** Researching methods, potentially guided by EEG feedback, to improve a dreamer's ability to remember and detail their dreams.
    * **Therapeutic Applications:** Exploring how AI-driven dream analysis could support psychological therapy, self-reflection, or managing recurring nightmares.
    * **More Advanced NLP Models:** Utilizing transformer-based language models (like BERT, GPT) fine-tuned specifically for dream language and symbolism.
    """)
    st.markdown("---")

    st.subheader("About the Project")
    st.write("""
        This project was developed as part of academic research into multimodal dream analysis. It leverages open-source data (DEED Dataset) and cutting-edge machine learning techniques to provide novel insights into the human dream experience. Our goal is to contribute to the scientific understanding of dreams and explore practical applications for dream interpretation and analysis.
    """)
    st.write(f"**Tools Used:** Python, Streamlit, Scikit-learn, MNE-Python, Antropy, Pandas, Matplotlib, Seaborn, Google Gemini AI SDK.")
    st.markdown("---")
    
    st.subheader("Disclaimer")
    st.warning("""
        **This application is for research and educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.** Dream interpretations and predictions generated by this tool are based on statistical models and AI patterns, and may not reflect individual psychological states or medical conditions. Always consult with a qualified healthcare provider for any health concerns.
    """)
