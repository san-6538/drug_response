import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_datasets
from src.data.merge_data import merge_datasets
from src.data.preprocess import preprocess_target
from src.features.feature_engineering import encode_interaction_features
from src.utils.encoders import encode_categorical

from src.training.train_random_forest import train_rf
from src.training.train_xgboost import train_xgb
from src.training.train_catboost import train_cat

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.confusion_matrix import plot_cm
from src.evaluation.feature_importance import plot_importance

def main():
    print("=== PharmGKB ML Pipeline ===")
    
    # 1. Load Data
    cv, ca, genes, variants, drugs, relationships, guidelines = load_datasets("data")
    
    if ca.empty and not cv.empty:
        print("Warning: clinical_annotations_cleaned.csv not found or empty. Using clinical_variants_cleaned.csv as base.")
        ca = cv.copy() # fallback for pipeline to run if data missing
    
    # 2. & 3. & 4. Merge Data
    df = merge_datasets(cv, ca, variants, drugs, guidelines)
    
    if df.empty:
        print("Fatal Error: Merged dataset is empty. Cannot continue training.")
        return
    
    # 6. Preprocess Rare Phenotypes
    df = preprocess_target(df)
    
    # 5. Feature Engineering
    df = encode_interaction_features(df)
    
    # Define features and target (Removing `recommendation` to prevent data leakage)
    target_col = "Phenotype"
    expected_cols = [
        "Gene", "Variant", "Drug", "Drug Class", "Variant_impact", 
        "guideline_exists", "evidence_level_num",
        "gene_drug", "variant_drug", "gene_variant"
    ]
    
    cat_cols = [
        "Gene", "Variant", "Drug", "Drug Class", "Variant_impact", 
        "gene_drug", "variant_drug", "gene_variant"
    ]
    
    features = [c for c in expected_cols if c in df.columns]
    actual_cat_cols = [c for c in cat_cols if c in df.columns]
    
    # 7. Encoding
    print("Encoding categorical features...")
    df, encoders, target_le = encode_categorical(df, actual_cat_cols, target_col)
    
    X = df[features]
    y = df[target_col]
    
    # 8. Split
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 8.5 Baseline Model
    from sklearn.dummy import DummyClassifier
    print("\nTraining Baseline Dummy Classifier...")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    
    # 9. Train Models
    models = {}
    models["Baseline (Majority)"] = baseline
    models["Random Forest"] = train_rf(X_train, y_train)
    models["XGBoost"] = train_xgb(X_train, y_train)
    models["CatBoost"] = train_cat(X_train, y_train)
    
    # 10. Evaluate & 11. Plot & Cross-Validation & SHAP
    print("\n=== Evaluation Results ===")
    
    final_metrics = {}
    
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        preds = model.predict(X_test)
        
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.ravel()
            
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            
        metrics = evaluate_predictions(name, y_test, preds, probs, model.classes_)
        plot_cm(y_test, preds, name, target_le)
        plot_importance(model, features, name)
        
        # 5-fold CV (Skip CatBoost if we want to save time, but it's okay to run it)
        print(f"Running 5-Fold Cross Validation for {name}...")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
            metrics["cv_f1_macro"] = scores.mean()
        except Exception as e:
            metrics["cv_f1_macro"] = 0.0
            print(f"CV Failed for {name}: {e}")
            
        final_metrics[name] = metrics

        # Generate SHAP strictly for XGBoost
        if name == "XGBoost":
            try:
                import shap
                import matplotlib.pyplot as plt
                print("Generating SHAP feature explanations...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_test) # Using the modern API for shap_values
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, show=False)
                os.makedirs("results/plots", exist_ok=True)
                plt.savefig("results/plots/xgboost_shap_summary.png", bbox_inches='tight')
                plt.close()
                print("SHAP plot saved to results/plots/xgboost_shap_summary.png")
            except Exception as e:
                print(f"Error generating SHAP values: {e}")
        
    print("\nFinal Model Summary:")
    for name, mets in final_metrics.items():
        print()
        print(name)
        print(f"Accuracy: {mets.get('acc', 0):.2f}")
        print(f"F1 Macro: {mets.get('f1_macro', 0):.2f}")
        print(f"Balanced Accuracy: {mets.get('bal_acc', 0):.2f}")
        print(f"CV F1 Macro: {mets.get('cv_f1_macro', 0):.2f}")
        if mets.get('top3_acc'):
            print(f"Top-3 Accuracy: {mets['top3_acc']:.2f}")

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
