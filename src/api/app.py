from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Pharmacogenomics ML Inference API", version="1.0")

# Load Models
MODEL_PATH = "models/xgboost_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    target_le = encoders.get("Phenotype")
else:
    model = None
    encoders = None
    target_le = None

class PatientInput(BaseModel):
    Gene: str
    Variant: str
    Drug: str
    Drug_Class: str = "Unknown"
    Variant_impact: str = "Unknown"
    guideline_exists: int = 0
    evidence_level_num: int = 0

@app.post("/predict")
def predict(data: PatientInput):
    if model is None:
        return {"error": "Model or encoders not found. Have you trained the models?"}

    # 1. Feature Engineering
    gene_drug = f"{data.Gene}*{data.Drug}"
    variant_drug = f"{data.Variant}*{data.Drug}"
    gene_variant = f"{data.Gene}_{data.Variant}"

    # 2. Build DataFrame
    input_data = {
        "Gene": [data.Gene],
        "Variant": [data.Variant],
        "Drug": [data.Drug],
        "Drug Class": [data.Drug_Class],
        "Variant_impact": [data.Variant_impact],
        "guideline_exists": [data.guideline_exists],
        "evidence_level_num": [data.evidence_level_num],
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
    
    for col in cat_cols:
        freq_dict = encoders.get(col, {})
        # Applying frequency encoding from dict. If not found, use 0.0
        df[col] = df[col].astype(str).map(freq_dict).fillna(0.0)
        
    # 4. Predict
    prediction_num = model.predict(df)[0]
    
    # XGBoost returns singular integers if ravelled, convert back safely
    if hasattr(prediction_num, "item"):
        prediction_num = prediction_num.item()
        
    try:
        prediction_label = target_le.inverse_transform([prediction_num])[0]
    except Exception:
        prediction_label = str(prediction_num) # Fallback
        
    return {
        "prediction": prediction_label, 
        "input": data.dict()
    }
