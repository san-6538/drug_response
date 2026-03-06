from catboost import CatBoostClassifier
import joblib
import os

def train_cat(X_train, y_train):
    print("Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        verbose=False,
        random_state=42,
        thread_count=-1
    )
    cat.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(cat, "models/catboost_model.pkl")
    return cat
