import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback

# --- APPLE SILICON HARDWARE OVERRIDE (THE FIX) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# Force TensorFlow to completely ignore the Mac GPU and use the CPU
tf.config.set_visible_devices([], 'GPU')
# -------------------------------------------------

from PIL import Image
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
st.set_page_config(page_title="Healthcare AI Dashboard", layout="wide", page_icon="🏥")
SAVE_DIR = "saved_models"
DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")

# --- CACHED PREPROCESSING UTILITY ---
@st.cache_data
def load_preprocessing_artifacts():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None
        
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    X = df.drop(columns=["diabetes"])
    # PANDAS WARNING FIX: Use exclude=["number"] to capture all string/categorical data safely
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    expected_columns = X_encoded.columns.tolist()
    
    scaler = StandardScaler()
    scaler.fit(X_encoded)
    
    return expected_columns, scaler, cat_cols, df

# --- SAFE MODEL LOADING ---
def load_trained_model(model_name):
    tf.keras.backend.clear_session()
    model_path = os.path.join(SAVE_DIR, f"{model_name}.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

# --- UI COMPONENTS ---
def render_dashboard():
    st.title("📊 Model Evaluation Dashboard")
    st.markdown("Explore the performance metrics and visualizations for your trained Deep Learning architectures.")
    
    comp_file = os.path.join(SAVE_DIR, "model_comparison.csv")
    if not os.path.exists(comp_file):
        st.warning("🚨 No models found. Please run your training scripts first.")
        return

    df = pd.read_csv(comp_file)
    st.header("🏆 Performance Comparison")
    st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'], color="#601EDB"))

    st.markdown("---")
    selected_model = st.selectbox("Select an Architecture to Inspect:", df["Model"].tolist())
    
    col1, col2 = st.columns(2)
    hist_plot = os.path.join(SAVE_DIR, f"{selected_model}_history.png")
    roc_plot = os.path.join(SAVE_DIR, f"{selected_model}_roc_pr.png")
    cm_plot = os.path.join(SAVE_DIR, f"{selected_model}_cm.png")

    # STREAMLIT WARNING FIX: Changed use_container_width=True to width="stretch"
    with col1:
        if os.path.exists(hist_plot): st.image(Image.open(hist_plot), width="stretch")
        if os.path.exists(cm_plot): st.image(Image.open(cm_plot), width="stretch")
            
    with col2:
        if os.path.exists(roc_plot): st.image(Image.open(roc_plot), width="stretch")
        report_path = os.path.join(SAVE_DIR, f"{selected_model}_report.txt")
        if os.path.exists(report_path):
            st.subheader("Classification Report")
            with open(report_path, "r") as f:
                st.text(f.read())

def render_prediction():
    st.title("🩺 Interactive Diabetes Prediction")
    st.markdown("Select a trained architecture and enter patient parameters to run a live diagnostic inference.")
    
    expected_columns, scaler, cat_cols, raw_df = load_preprocessing_artifacts()
    
    if expected_columns is None or raw_df is None:
        st.error(f"Dataset not found at {DATA_PATH}. Cannot generate input features.")
        return

    comp_file = os.path.join(SAVE_DIR, "model_comparison.csv")
    if not os.path.exists(comp_file):
        st.warning("Please train your models first.")
        return
        
    df = pd.read_csv(comp_file)
    available_models = df["Model"].tolist()
    
    st.header("1. Model Selection")
    chosen_model_name = st.selectbox("Choose the Neural Network for inference:", available_models)

    st.markdown("---")
    st.header("2. Patient Data Entry")
    
    with st.form("prediction_form"):
        colA, colB, colC = st.columns(3)
        
        input_data = {}
        with colA:
            input_data['gender'] = st.selectbox("Gender", raw_df['gender'].unique())
            input_data['age'] = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
            input_data['hypertension'] = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
            
        with colB:
            input_data['heart_disease'] = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
            input_data['smoking_history'] = st.selectbox("Smoking History", raw_df['smoking_history'].unique())
            input_data['bmi'] = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)
            
        with colC:
            input_data['HbA1c_level'] = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
            input_data['blood_glucose_level'] = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120)

        submit = st.form_submit_button("Run Diagnostic Inference", type="primary")

    if submit:
        try:
            model = load_trained_model(chosen_model_name)
            if model is None:
                st.error(f"Could not load {chosen_model_name}.keras. Check your saved_models folder.")
                return

            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False, dtype=float)
            input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0.0)
            
            if scaler is None:
                st.error("Scaler could not be loaded. Check your data preprocessing.")
                return
                
            input_scaled = scaler.transform(input_aligned).astype(np.float32)

            with st.spinner(f"Analyzing patient vitals using {chosen_model_name}..."):
                probs = model.predict(input_scaled, verbose=0)[0]
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class] * 100)

            st.markdown("---")
            st.header("📝 Inference Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred_class == 1:
                    st.error(f"### Prediction: **Positive (Diabetic)**\nConfidence: **{confidence:.2f}%**")
                else:
                    st.success(f"### Prediction: **Negative (Non-Diabetic)**\nConfidence: **{confidence:.2f}%**")
                    
            with res_col2:
                st.write(f"**Probability Scores ({chosen_model_name}):**")
                st.progress(float(probs[1]), text=f"Positive Probability: {probs[1]*100:.1f}%")
                st.progress(float(probs[0]), text=f"Negative Probability: {probs[0]*100:.1f}%")
                
        except Exception as e:
            st.error("🚨 An error occurred during prediction.")
            st.code(traceback.format_exc())

# --- MAIN APP ROUTING ---
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Model Dashboard", "Interactive Prediction"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("**System Status:**\nRunning Centralized Baseline Models (CPU Mode Enabled).")

    if page == "Model Dashboard":
        render_dashboard()
    elif page == "Interactive Prediction":
        render_prediction()

if __name__ == "__main__":
    main()