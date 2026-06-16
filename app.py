import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
st.set_page_config(page_title="Federated Healthcare AI", layout="wide", page_icon="🏥")
SAVE_DIR = "saved_models"
DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")

# --- CACHED PREPROCESSING UTILITY ---
# This ensures user inputs match the exact feature shape and scaling used during training
@st.cache_data
def load_preprocessing_artifacts():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None
        
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    X = df.drop(columns=["diabetes"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Get the exact dummy columns the model was trained on
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    expected_columns = X_encoded.columns.tolist()
    
    # Fit the scaler on the full dataset
    scaler = StandardScaler()
    scaler.fit(X_encoded)
    
    return expected_columns, scaler, cat_cols, df

# --- UI COMPONENTS ---
def render_dashboard():
    st.title("📊 Federated Model Dashboard")
    st.markdown("Explore centralized architectures and the privacy-preserving federated global model.")
    
    comp_file = os.path.join(SAVE_DIR, "model_comparison.csv")
    if not os.path.exists(comp_file):
        st.warning("🚨 No models found. Please run your training scripts first.")
        return

    df = pd.read_csv(comp_file)
    
    # Highlight federated vs centralized
    st.header("🏆 Performance Comparison")
    st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'], color='#a8e6cf'))

    # Architecture deep dive
    st.markdown("---")
    selected_model = st.selectbox("Select an Architecture to Inspect:", df["Model"].tolist())
    
    col1, col2 = st.columns(2)
    hist_plot = os.path.join(SAVE_DIR, f"{selected_model}_history.png")
    roc_plot = os.path.join(SAVE_DIR, f"{selected_model}_roc_pr.png")
    cm_plot = os.path.join(SAVE_DIR, f"{selected_model}_cm.png")
    prog_plot = os.path.join(SAVE_DIR, "federated_progression.png")

    if selected_model == "federated_global_model" and os.path.exists(prog_plot):
        st.image(Image.open(prog_plot), caption="Federated Learning Round Progression", use_container_width=True)

    with col1:
        if os.path.exists(hist_plot): st.image(Image.open(hist_plot), use_container_width=True)
        elif selected_model == "federated_global_model": st.info("History plot not applicable for aggregated Federated Model.")
        
        if os.path.exists(cm_plot): st.image(Image.open(cm_plot), use_container_width=True)
            
    with col2:
        if os.path.exists(roc_plot): st.image(Image.open(roc_plot), use_container_width=True)
        
        report_path = os.path.join(SAVE_DIR, f"{selected_model}_report.txt")
        if os.path.exists(report_path):
            st.subheader("Classification Report")
            with open(report_path, "r") as f:
                st.text(f.read())

def render_prediction():
    st.title("🩺 Interactive Diabetes Prediction")
    st.markdown("Enter patient parameters below to securely run inference against the trained Neural Network.")
    
    expected_columns, scaler, cat_cols, raw_df = load_preprocessing_artifacts()
    
    if expected_columns is None:
        st.error(f"Dataset not found at {DATA_PATH}. Cannot generate input features.")
        return

    # Dynamically build UI based on the real dataset features
    st.header("Patient Data Entry")
    
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
            input_data['HbA1c_level'] = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)
            input_data['blood_glucose_level'] = st.number_input("Blood Glucose Level", min_value=50, max_value=350, value=120)

        submit = st.form_submit_button("Predict with Global Model", type="primary")

    if submit:
        # 1. Load the Federated Model (or fallback to best available)
        model_path = os.path.join(SAVE_DIR, "federated_global_model.keras")
        if not os.path.exists(model_path):
            # Fallback logic if FL hasn't run yet
            metrics = pd.read_csv(os.path.join(SAVE_DIR, "model_comparison.csv"))
            best_model = metrics.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
            model_path = os.path.join(SAVE_DIR, f"{best_model}.keras")
            st.warning(f"Federated model not found. Defaulting to best central model: {best_model}")

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        # 2. Preprocess Input Exactly as Training
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False, dtype=float)
        
        # Align columns with training data (fill missing dummies with 0)
        input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0.0)
        
        # Scale
        input_scaled = scaler.transform(input_aligned)

        # 3. Inference
        with st.spinner("Processing..."):
            probs = model.predict(input_scaled)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class] * 100

        # 4. Display Results
        st.markdown("---")
        st.header("📝 Inference Results")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred_class == 1:
                st.error(f"### Prediction: **Positive (Diabetic)**\nConfidence: **{confidence:.2f}%**")
            else:
                st.success(f"### Prediction: **Negative (Non-Diabetic)**\nConfidence: **{confidence:.2f}%**")
                
        with res_col2:
            st.write("**Probability Scores:**")
            st.progress(float(probs[1]), text=f"Positive Probability: {probs[1]*100:.1f}%")
            st.progress(float(probs[0]), text=f"Negative Probability: {probs[0]*100:.1f}%")

# --- MAIN APP ROUTING ---
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # Generic medical icon
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Model Dashboard", "Interactive Prediction"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Privacy Note:**\nThis system aggregates weights via FedAvg. Raw patient data never leaves the local environment.")

    if page == "Model Dashboard":
        render_dashboard()
    elif page == "Interactive Prediction":
        render_prediction()

if __name__ == "__main__":
    main()