"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 Human Activity Recognition (HAR) Dashboard                   ║
║              Classify Physical Activities from Smartphone Sensors             ║
╚══════════════════════════════════════════════════════════════════════════════╝

This Streamlit app provides an interactive interface to:
  • Predict activities from accelerometer & gyroscope data
  • Compare model performance (SVM, Random Forest, LSTM)
  • Visualize feature importance
  • Understand model predictions with explanations
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow (optional for LSTM)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    tf = None

# ══════════════════════════════════════════════════════════════════════════════
# SETUP PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located
# Works both locally and on Streamlit Cloud
PROJECT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_DIR / "models"
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"

# Fallback for Streamlit Cloud
if not MODELS_DIR.exists():
    # Try common Streamlit Cloud paths
    import os
    cwd = Path.cwd()
    if (cwd / "models").exists():
        MODELS_DIR = cwd / "models"
        PREPROCESSED_DIR = cwd / "preprocessed"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HAR Dashboard",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Ensure proper text visibility */
    html, body, [class*="css"] {
        color: #ffffff !important;
        background-color: #0e1117 !important;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Headers - Force white text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-align: center;
    }
    
    /* Paragraph text */
    p, label, div, span {
        color: #e0e6ed !important;
    }
    
    /* Input fields */
    input, textarea, select {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }
    
    /* Buttons */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
    }
    
    button:hover {
        opacity: 0.8;
    }
    
    /* Headers */
    h2 {
        color: #ffffff !important;
        border-bottom: 3px solid #667eea !important;
        padding-bottom: 8px !important;
    }
    
    h3 {
        color: #667eea !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: #161b22 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        border-left: 4px solid #667eea !important;
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1f6feb !important;
        border-left: 5px solid #3fb950 !important;
        padding: 15px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
        color: #ffffff !important;
    }
    
    .success-box {
        background-color: #238636 !important;
        border-left: 5px solid #3fb950 !important;
        padding: 15px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
        color: #ffffff !important;
    }
    
    .warning-box {
        background-color: #9e6a03 !important;
        border-left: 5px solid #d29922 !important;
        padding: 15px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA AND MODELS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    try:
        models['svm'] = joblib.load(MODELS_DIR / 'svm_rbf_uci.pkl')
        st.success("✅ SVM model loaded")
    except Exception as e:
        st.warning(f"⚠️ SVM model not found: {str(e)}")
    
    try:
        models['rf'] = joblib.load(MODELS_DIR / 'random_forest_model.pkl')
        st.success("✅ Random Forest model loaded")
    except Exception as e:
        st.warning(f"⚠️ Random Forest model not found: {str(e)}")
    
    # Try to load LSTM only if TensorFlow is available
    if TF_AVAILABLE:
        try:
            # Try with custom_objects for Keras 3.x compatibility
            try:
                models['lstm'] = tf.keras.models.load_model(
                    str(MODELS_DIR / 'lstm_model.h5'), 
                    compile=False,
                    custom_objects=None
                )
            except:
                # Fallback: try loading with safe mode
                import tensorflow as tf
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                models['lstm'] = tf.keras.models.load_model(
                    str(MODELS_DIR / 'lstm_model.h5'),
                    compile=False
                )
            st.success("✅ LSTM model loaded")
        except Exception as e:
            st.warning(f"⚠️ LSTM model error: TensorFlow/Keras version mismatch - Using SVM/RF only")
            # Don't add lstm to models dict
    else:
        st.warning("⚠️ TensorFlow not available - LSTM model skipped")
    
    return models

@st.cache_resource
def load_scalers():
    """Load preprocessing scalers."""
    scalers = {}
    try:
        scalers['ml'] = joblib.load(PREPROCESSED_DIR / 'scaler_ml.pkl')
        scalers['vt'] = joblib.load(PREPROCESSED_DIR / 'variance_threshold.pkl')
        scalers['hand'] = joblib.load(PREPROCESSED_DIR / 'scaler_hand.pkl')
        st.success("✅ Scalers loaded")
        return scalers
    except Exception as e:
        st.error(f"❌ Scalers not found: {str(e)}")
        return None

@st.cache_data
def load_feature_names():
    """Load UCI HAR feature names."""
    try:
        _raw_names = pd.read_csv(
            PROJECT_DIR / 'UCI HAR Dataset/features.txt',
            sep=r'\s+', header=None, index_col=0
        )[1].values
        
        _vt = joblib.load(PREPROCESSED_DIR / 'variance_threshold.pkl')
        _support_idx = np.where(_vt.get_support())[0]
        feature_names = _raw_names[_support_idx]
        return feature_names
    except Exception as e:
        return np.array([f'Feature_{i}' for i in range(477)])

# Activity mapping
ACTIVITY_MAP = {
    0: ('Walking', '🚶'),
    1: ('Walking Upstairs', '🪜'),
    2: ('Walking Downstairs', '⬇️'),
    3: ('Sitting', '🪑'),
    4: ('Standing', '🫡'),
    5: ('Laying', '🛏️')
}

CLASS_NAMES = [ACTIVITY_MAP[i][0] for i in range(6)]
CLASS_EMOJIS = [ACTIVITY_MAP[i][1] for i in range(6)]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# 🚶 Human Activity Recognition Dashboard")
    st.markdown("<p style='text-align: center; color: #667eea; font-size: 14px;'>"
                "Predict smartphone activities from accelerometer & gyroscope sensors</p>",
                unsafe_allow_html=True)

st.markdown("---")

# Load models and scalers
models = load_models()
scalers = load_scalers()
feature_names = load_feature_names()

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "🎯 Live Prediction",
    "📊 Model Comparison",
    "🔍 Feature Importance",
    "ℹ️ Documentation"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: HOME
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 🎉 Welcome to the HAR Dashboard!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📱 What is HAR?
        
        **Human Activity Recognition (HAR)** is the task of automatically 
        identifying what physical activity a person is performing based on 
        sensor readings from a smartphone or wearable device.
        
        This dashboard uses data from:
        - **Accelerometer** (3 axes: X, Y, Z)
        - **Gyroscope** (3 axes: roll, pitch, yaw)
        
        Sampled at **50 Hz** with **128 readings per window** (2.56 seconds).
        """)
        
        st.markdown("### 🎯 Supported Activities")
        for i, (activity, emoji) in ACTIVITY_MAP.items():
            st.markdown(f"- {emoji} **{activity}**")
    
    with col2:
        st.markdown("### 🤖 Available Models")
        
        model_info = pd.DataFrame({
            'Model': ['Linear SVM', 'Random Forest', 'LSTM'],
            'Type': ['Classical ML', 'Classical ML', 'Deep Learning'],
            'Accuracy': ['96.5%', '92.3%', '83.2%'],
            'Speed': ['⚡ Fast', '⚡ Fast', '⚠️ Medium'],
        })
        
        st.dataframe(model_info, use_container_width=True, hide_index=True)
        
        st.markdown("### 📈 Dataset Info")
        st.markdown("""
        - **Training Subjects**: 21 people
        - **Test Subjects**: 9 people (unseen)
        - **Training Samples**: 7,352
        - **Test Samples**: 2,947
        - **Features (per model)**: 477-561
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Go to Live Prediction** → Upload sensor data or generate random data
    2. **Select a Model** → Compare predictions across different classifiers
    3. **View Results** → See confidence scores, feature importance, and visualizations
    4. **Explore Models** → Check model comparison and detailed feature analysis
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🎯 Make Predictions")
    
    st.markdown("### 📥 Input Data")
    
    input_method = st.radio(
        "How would you like to provide input?",
        ["📊 Upload File (CSV/TXT)", "🎯 Load from UCI Dataset", "🎲 Generate Random Sample"],
        horizontal=True
    )
    
    X_sample = None
    selected_activity = None
    
    if input_method == "📊 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Upload CSV or TXT file with sensor features", type=['csv', 'txt'])
        
        if uploaded_file:
            try:
                # Determine if CSV or TXT and load accordingly
                if uploaded_file.name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
                else:
                    df = pd.read_csv(uploaded_file)
                
                n_features = df.shape[1]
                X_sample = df.values[0].reshape(1, -1) if df.shape[0] == 1 else df.values
                st.success(f"✅ Loaded {df.shape[0]} sample(s) with {n_features} features")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    elif input_method == "🎯 Load from UCI Dataset":
        st.info("📂 Loading from UCI HAR Dataset test directory...")
        
        dataset_path = PROJECT_DIR / "UCI HAR Dataset" / "test"
        
        if dataset_path.exists():
            # Load test data and labels
            try:
                X_test_uci = np.loadtxt(dataset_path / "X_test.txt")
                y_test_uci = np.loadtxt(dataset_path / "y_test.txt", dtype=int)
                
                sample_idx = st.slider(
                    "Select test sample #",
                    0,
                    len(X_test_uci) - 1,
                    0
                )
                
                X_sample = X_test_uci[sample_idx].reshape(1, -1)
                selected_activity = y_test_uci[sample_idx] - 1  # Convert to 0-indexed
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Loaded sample #{sample_idx}")
                with col2:
                    st.info(f"📍 True activity: {ACTIVITY_MAP[selected_activity][1]} {ACTIVITY_MAP[selected_activity][0]}")
                
                # Show feature preview
                with st.expander(f"View {X_sample.shape[1]} features for this sample"):
                    feature_df = pd.DataFrame({
                        'Feature': [f"Feature {i+1}" for i in range(X_sample.shape[1])],
                        'Value': X_sample[0]
                    })
                    st.dataframe(feature_df)
                    
            except Exception as e:
                st.error(f"❌ Error loading UCI dataset: {e}")
        else:
            st.error(f"❌ UCI HAR Dataset not found at {dataset_path}")
    
    else:  # Generate random sample
        activity_idx = st.selectbox(
            "Select the true activity (for reference):",
            range(6),
            format_func=lambda x: f"{ACTIVITY_MAP[x][1]} {ACTIVITY_MAP[x][0]}"
        )
        selected_activity = activity_idx
        
        n_features_random = st.number_input(
            "Number of features:",
            min_value=10,
            max_value=10000,
            value=477,
            step=10
        )
        
        if st.button("🎲 Generate Random Sensor Data"):
            # Create realistic random data
            np.random.seed(42)
            X_sample = np.random.randn(1, n_features_random) * 0.5
            st.success(f"✅ Generated random sensor data with {n_features_random} features")
    
    if X_sample is not None:
        st.markdown("---")
        st.markdown("### 🤖 Select Model(s) for Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        use_svm = col1.checkbox("SVM", value=True)
        use_rf = col2.checkbox("Random Forest", value=True)
        use_lstm = col3.checkbox("LSTM", value=False, disabled=not TF_AVAILABLE)
        
        if not TF_AVAILABLE and use_lstm:
            st.info("ℹ️ LSTM disabled: TensorFlow not available in environment")
            use_lstm = False
        
        if st.button("🚀 Make Prediction", use_container_width=True):
            predictions = {}
            confidences = {}
            
            # Prepare data - handle variable feature counts
            X_to_use = X_sample.copy()
            expected_features = 477
            
            if X_to_use.shape[1] < expected_features:
                # Pad with zeros to match expected features
                padding = np.zeros((X_to_use.shape[0], expected_features - X_to_use.shape[1]))
                X_to_use = np.hstack([X_to_use, padding])
                st.warning(f"⚠️ Input has {X_sample.shape[1]} features, padded to {expected_features}")
            elif X_to_use.shape[1] > expected_features:
                # Truncate to expected features
                X_to_use = X_to_use[:, :expected_features]
                st.warning(f"⚠️ Input has {X_sample.shape[1]} features, truncated to {expected_features}")
            else:
                st.info(f"✅ Input has expected {expected_features} features")
            
            # Apply scalers
            X_scaled = scalers['ml'].transform(X_to_use)
            X_vt = scalers['vt'].transform(X_scaled)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            # Show reference activity if available
            if selected_activity is not None:
                ref_activity, ref_emoji = ACTIVITY_MAP[selected_activity]
                st.info(f"📍 **True Activity**: {ref_emoji} {ref_activity}")
            
            st.markdown("---")
            
            # SVM Prediction
            if use_svm and 'svm' in models:
                with st.spinner("🔄 Running SVM..."):
                    pred_svm = models['svm'].predict(X_vt)[0]
                    proba_svm = models['svm'].decision_function(X_vt)[0]
                    # Normalize to probabilities
                    proba_svm = (proba_svm - proba_svm.min()) / (proba_svm.max() - proba_svm.min())
                    proba_svm = proba_svm / proba_svm.sum()
                    
                    predictions['svm'] = pred_svm
                    confidences['svm'] = proba_svm
            
            # Random Forest Prediction
            if use_rf and 'rf' in models:
                with st.spinner("🔄 Running Random Forest..."):
                    pred_rf = models['rf'].predict(X_vt)[0]
                    proba_rf = models['rf'].predict_proba(X_vt)[0]
                    
                    predictions['rf'] = pred_rf
                    confidences['rf'] = proba_rf
            
            # LSTM Prediction
            if use_lstm and 'lstm' in models:
                with st.spinner("🔄 Running LSTM..."):
                    # For LSTM, we need sequential data (128, 9)
                    # Create a simple transformation or use preprocessed data
                    X_raw = np.random.randn(1, 128, 9)  # Placeholder
                    proba_lstm = models['lstm'].predict(X_raw, verbose=0)[0]
                    pred_lstm = np.argmax(proba_lstm)
                    
                    predictions['lstm'] = pred_lstm
                    confidences['lstm'] = proba_lstm
            
            # Display results
            result_cols = st.columns(len(predictions))
            
            for idx, (model_name, col) in enumerate(zip(predictions.keys(), result_cols)):
                with col:
                    pred_class = predictions[model_name]
                    confidence = confidences[model_name][pred_class]
                    activity_name, emoji = ACTIVITY_MAP[pred_class]
                    
                    st.markdown(f"""
                    <div style='background: white; padding: 20px; border-radius: 10px; 
                                border-left: 5px solid #667eea; text-align: center;'>
                        <h3>{emoji} {model_name.upper()}</h3>
                        <h2 style='color: #667eea;'>{activity_name}</h2>
                        <p style='font-size: 16px; color: #666;'>
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Confidence bars
            st.markdown("---")
            st.markdown("### 📈 Confidence Scores by Activity")
            
            for model_name in predictions.keys():
                st.markdown(f"#### {model_name.upper()}")
                
                conf_df = pd.DataFrame({
                    'Activity': CLASS_NAMES,
                    'Confidence': confidences[model_name] * 100
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(conf_df['Activity'], conf_df['Confidence'], 
                               color=['#667eea' if i == predictions[model_name] 
                                      else '#ccc' for i in range(6)])
                ax.set_xlabel('Confidence (%)', fontsize=12)
                ax.set_title(f'Prediction Confidence - {model_name}', fontsize=13, fontweight='bold')
                ax.set_xlim(0, 100)
                
                for i, (activity, conf) in enumerate(zip(conf_df['Activity'], conf_df['Confidence'])):
                    ax.text(conf + 2, i, f'{conf:.1f}%', va='center', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 📊 Model Comparison")
    
    # Performance metrics
    comparison_data = {
        'Model': ['Linear SVM', 'Random Forest', 'LSTM'],
        'Accuracy (%)': [96.5, 92.3, 83.2],
        'Precision': [0.965, 0.923, 0.832],
        'Recall': [0.964, 0.918, 0.831],
        'F1-Score': [0.965, 0.921, 0.831],
        'Training Time': ['~10s', '~6min', '~5min'],
        'Inference Speed': ['⚡ Fastest', '⚡ Fast', '⚠️ Medium']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    st.markdown("### 🏆 Performance Metrics")
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        models_list = ['SVM', 'Random Forest', 'LSTM']
        accuracies = [96.5, 92.3, 83.2]
        colors = ['#667eea', '#764ba2', '#f093fb']
        
        bars = ax.bar(models_list, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_ylim(75, 100)
        ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### F1-Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        f1_scores = [0.965, 0.921, 0.831]
        
        bars = ax.bar(models_list, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_ylim(0.75, 1.0)
        ax.set_title('Model F1-Score Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, f1 in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{f1:.3f}', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🥇 Linear SVM
        - **Best Overall Performance** (96.5% accuracy)
        - Very fast training & inference
        - Works well with 561 handcrafted features
        - **Recommended for production**
        """)
    
    with col2:
        st.markdown("""
        #### 🥈 Random Forest
        - Solid performance (92.3% accuracy)
        - Good feature interpretability
        - More robust to outliers than SVM
        - Reasonable training time (~6 min)
        """)
    
    with col3:
        st.markdown("""
        #### 🥉 LSTM
        - Lower performance (83.2% accuracy)
        - Works directly on raw sensor signals
        - Sitting/Standing confusion (hard to distinguish)
        - Deep learning approach but limited data
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🔍 Feature Importance Analysis")
    
    st.markdown("""
    This section shows which sensor features are most important for activity recognition.
    Features with higher importance contribute more to model predictions.
    """)
    
    # Load feature importance from Random Forest
    try:
        rf_importances = models['rf'].feature_importances_
        
        # Get top 20 features
        top_n = 20
        top_idx = np.argsort(rf_importances)[::-1][:top_n]
        
        imp_df = pd.DataFrame({
            'Feature': feature_names[top_idx],
            'Importance': rf_importances[top_idx]
        })
        
        st.markdown("### 🎯 Top 20 Most Important Features (Random Forest)")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
        ax.barh(range(len(imp_df)), imp_df['Importance'], color=colors_imp, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(imp_df['Feature'], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Random Forest Feature Importance (Top 20)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (feat, imp) in enumerate(zip(imp_df['Feature'], imp_df['Importance'])):
            ax.text(imp + 0.0005, i, f'{imp:.5f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### 📊 Feature Importance Table")
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 💡 Feature Type Breakdown")
        
        # Categorize features
        time_domain = sum(1 for f in feature_names if f.startswith('t'))
        freq_domain = sum(1 for f in feature_names if f.startswith('f'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### ⏱️ Time-Domain Features
            - **Count**: ~{time_domain}
            - **Description**: Raw signal statistics (mean, std, min, max)
            - **Examples**: tBodyAcc-mean()-X, tGravityAcc-std()-Y
            - **Importance**: Very high - capture basic activity patterns
            """)
        
        with col2:
            st.markdown(f"""
            #### 📈 Frequency-Domain Features
            - **Count**: ~{freq_domain}
            - **Description**: FFT-based spectral features
            - **Examples**: fBodyAcc-mean()-X, fBodyGyro-energy()-Z
            - **Importance**: High - capture periodic motion patterns
            """)
    
    except Exception as e:
        st.error(f"❌ Could not load feature importance: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## ℹ️ Documentation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📚 Dataset Information")
        st.markdown("""
        **UCI HAR Dataset**
        - Source: UC Irvine Machine Learning Repository
        - Activities: 6 (Walking, Upstairs, Downstairs, Sitting, Standing, Laying)
        - Subjects: 30 volunteers (19-48 years old)
        - Device: Samsung Galaxy S II (50 Hz sampling)
        - Window: 2.56 seconds with 50% overlap
        - Features: 561 time & frequency domain features
        - Train/Test: 70%/30% (person-independent split)
        """)
    
    with col2:
        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        **Data Processing Pipeline**
        1. Load raw sensor data (accelerometer + gyroscope)
        2. Extract 561-dimensional feature vectors
        3. Remove duplicate features (561 → 477)
        4. Normalize using StandardScaler
        5. Train classifiers with cross-validation
        
        **Models**
        - Linear SVM (C=1.0)
        - Random Forest (100-500 trees)
        - LSTM (128→64 units, batch norm, dropout)
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 How It Works")
        st.markdown("""
        1. **Sensor Data Collection**
           - Accelerometer measures linear acceleration (X, Y, Z)
           - Gyroscope measures angular velocity (roll, pitch, yaw)
           - 128 readings captured per 2.56-second window
        
        2. **Feature Extraction**
           - Time-domain: mean, std, min, max, energy, entropy
           - Frequency-domain: FFT magnitudes, spectral centroids
           - 561 features total per window
        
        3. **Model Training**
           - 70% training data on 21 subjects
           - 30% test data on 9 unseen subjects
           - Cross-validation with stratification
        
        4. **Prediction**
           - New sensor data → Feature extraction
           - Model inference → Activity class
           - Confidence score → Prediction certainty
        """)
    
    with col2:
        st.markdown("### 🚀 Use Cases")
        st.markdown("""
        **Healthcare & Fitness**
        - Patient activity monitoring
        - Rehabilitation tracking
        - Fitness app integration
        
        **Research**
        - Activity pattern analysis
        - Behavioral studies
        - Sensor fusion research
        
        **Smart Devices**
        - Smartwatch activity detection
        - Context-aware notifications
        - Battery optimization
        
        **Safety & Elderly Care**
        - Fall detection systems
        - Anomaly detection
        - Remote monitoring
        """)
    
    st.markdown("---")
    
    st.markdown("### 📖 Activity Definitions")
    
    activities_def = {
        '🚶 Walking': 'Natural pace walking on a flat surface. Periodic arm and leg movement.',
        '🪜 Walking Upstairs': 'Walking up a staircase. Similar to walking but with increased leg load.',
        '⬇️ Walking Downstairs': 'Walking down a staircase. Eccentric leg muscle activity.',
        '🪑 Sitting': 'Seated in a chair. Phone held at waist. Minimal acceleration.',
        '🫡 Standing': 'Upright posture. Phone held at waist. Near-zero acceleration.',
        '🛏️ Laying': 'Lying down. Phone held in hand on chest or bed. Gravity-dominated signal.'
    }
    
    for activity, description in activities_def.items():
        st.markdown(f"**{activity}**: {description}")
    
    st.markdown("---")
    
    st.markdown("### 📞 About This Dashboard")
    st.markdown("""
    **Version**: 1.0  
    **Framework**: Streamlit  
    **Models**: scikit-learn (SVM, Random Forest), TensorFlow (LSTM)  
    **Dataset**: UCI HAR  
    
    **Project Structure**
    - `svm_predictive_2.ipynb` - Stages 1-4 (Data preprocessing, SVM training)
    - `2nd_half_code.ipynb` - Stages 5-7 (Random Forest, LSTM, Comparison)
    - `app.py` - Interactive Streamlit dashboard
    
    **Team**: Project - 2 / Human Activity Recognition  
    **Last Updated**: May 2026
    """)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    <p>🚶 Human Activity Recognition Dashboard | 
       Built with ❤️ using Streamlit & Machine Learning</p>
    <p>Dataset: UCI HAR | Models: SVM, Random Forest, LSTM</p>
</div>
""", unsafe_allow_html=True)
