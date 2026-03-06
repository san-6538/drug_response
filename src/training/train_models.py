from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {}

    print("\n--- Training Models ---")

    # --- 1. Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # --- 2. XGBoost ---
    print("\nTraining XGBoost...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    models["XGBoost"] = xgb

    # --- 3. CatBoost ---
    print("\nTraining CatBoost...")
    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        verbose=100,
        random_state=42,
        thread_count=-1
    )
    cat.fit(X_train, y_train)
    models["CatBoost"] = cat

    print("\n--- Training Finished ---")

    os.makedirs("models", exist_ok=True)
    joblib.dump(cat, "models/catboost_model.pkl")
    joblib.dump(xgb, "models/xgboost_model.pkl")

    return models, X_test, y_test
