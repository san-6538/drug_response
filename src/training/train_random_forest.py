from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_rf(X_train, y_train):
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest_model.pkl")
    return rf
