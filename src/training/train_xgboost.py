from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os

def train_xgb(X_train, y_train):
    print("Training XGBoost...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgboost_model.pkl")
    return xgb
