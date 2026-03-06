from sklearn.preprocessing import LabelEncoder
import joblib
import os

def encode_categorical(df, cat_cols, target_col):
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            # Frequency encoding
            freq = df[col].astype(str).fillna("Unknown").value_counts(normalize=True).to_dict()
            df[col] = df[col].astype(str).map(freq).fillna(0)
            encoders[col] = freq
            
    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col].astype(str))
    encoders[target_col] = target_le
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")
    
    return df, encoders, target_le
