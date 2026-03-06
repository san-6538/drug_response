import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Configure page
st.set_page_config(page_title="PharmGKB Predictor", layout="wide")
st.title("💊 Pharmacogenomics Clinical Predictor")
st.markdown("Predict patient drug response phenotypes using genetic interaction networks and PharmGKB guidelines.")

# Paths
MODEL_PATH = "models/xgboost_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
RESULTS_PATH = "results/prediction_history.csv"

# Load models safely
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        target_le = encoders.get("Phenotype")
        return model, encoders, target_le
    return None, None, None

model, encoders, target_le = load_assets()

if model is None:
    st.error("Model or encoders not found! Please run the training pipeline first (`python main.py`).")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Data")
    gene = st.text_input("Gene Symbol", value="CYP2C9")
    variant = st.text_input("Variant ID", value="rs1799853")
    drug = st.text_input("Drug Name", value="Warfarin")
    drug_class = st.text_input("Drug Class", value="Anticoagulant")
    variant_impact = st.selectbox("Variant Impact", ["Unknown", "Pathogenic", "Benign", "Likely Pathogenic"])

with col2:
    st.header("Clinical Guidelines")
    guideline_exists = st.radio("Is there a known CPIC/PharmGKB guideline?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    evidence_level_num = st.slider("Evidence Level (0=None, 1=C, 2=B, 3=A)", 0, 3, 3)

predict_btn = st.button("Predict Clinical Category", type="primary")

if predict_btn:
    # 1. Feature Engineering
    gene_drug = f"{gene}*{drug}"
    variant_drug = f"{variant}*{drug}"
    gene_variant = f"{gene}_{variant}"

    # 2. Build DataFrame
    input_data = {
        "Gene": [gene],
        "Variant": [variant],
        "Drug": [drug],
        "Drug Class": [drug_class],
        "Variant_impact": [variant_impact],
        "guideline_exists": [guideline_exists],
        "evidence_level_num": [evidence_level_num],
        "gene_drug": [gene_drug],
        "variant_drug": [variant_drug],
        "gene_variant": [gene_variant]
    }
    
    df = pd.DataFrame(input_data)
    
    # 3. Apply Encoders
    cat_cols = [
        "Gene", "Variant", "Drug", "Drug Class", "Variant_impact", 
        "gene_drug", "variant_drug", "gene_variant"
    ]
    
    encoded_df = df.copy()
    for col in cat_cols:
        freq_dict = encoders.get(col, {})
        encoded_df[col] = encoded_df[col].astype(str).map(freq_dict).fillna(0.0)
        
    # 4. Predict
    with st.spinner("Analyzing genetic interactions..."):
        prediction_num = model.predict(encoded_df)[0]
        
        if hasattr(prediction_num, "item"):
            prediction_num = prediction_num.item()
            
        try:
            prediction_label = target_le.inverse_transform([prediction_num])[0]
        except Exception:
            prediction_label = str(prediction_num)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(encoded_df)[0]
            top3_idx = probs.argsort()[-3:][::-1]
            try:
                top3_labels = target_le.inverse_transform(top3_idx)
                top3_probs = probs[top3_idx]
            except Exception:
                top3_labels = [str(i) for i in top3_idx]
                top3_probs = probs[top3_idx]
        else:
            top3_labels, top3_probs = None, None

    # UI Output
    st.success(f"### Predicted Outcome: **{prediction_label}**")
    
    if top3_labels is not None:
        st.write("#### Top 3 Probabilities:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}**: {prob:.2%}")
            
    # 5. Store Prediction
    log_data = input_data.copy()
    log_data["Predicted_Phenotype"] = [prediction_label]
    log_data["Timestamp"] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    
    log_df = pd.DataFrame(log_data)
    
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(RESULTS_PATH):
        log_df.to_csv(RESULTS_PATH, index=False)
    else:
        log_df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
        
    st.info(f"Input and prediction saved securely to `{RESULTS_PATH}`.")

# Optional: View History
st.divider()
if st.checkbox("Show Prediction History"):
    if os.path.exists(RESULTS_PATH):
        history_df = pd.read_csv(RESULTS_PATH)
        st.dataframe(history_df.tail(10)) # Show last 10 entries
    else:
        st.write("No predictions logged yet.")
