from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, top_k_accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_feature_importance(model, feature_names, name):
    importance = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    
    # Sort features by importance
    idx = np.argsort(importance)[::-1]
    sorted_importance = importance[idx]
    sorted_features = np.array(feature_names)[idx]
    
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis", hue=sorted_features, legend=False)
    plt.title(f"{name} Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"results/{filename}_feature_importance.png")
    print(f"\nFeature importance chart saved to results/{filename}_feature_importance.png")


def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        predictions = model.predict(X_test)
        
        # Flatten predictions for CatBoost if necessary
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        acc = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average="macro")
        bal_acc = balanced_accuracy_score(y_test, predictions)
        
        # Top-3 Accuracy
        top3_acc = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            try:
                top3_acc = top_k_accuracy_score(y_test, probs, k=3, labels=model.classes_)
            except Exception as e:
                print(f"Could not calculate Top-3 accuracy for {name}: {e}")

        print("\n" + "="*40)
        print(f" {name}")
        print("="*40)
        print(f"Accuracy:          {acc:.4f}")
        print(f"F1 Score (Macro):  {f1_macro:.4f}")
        print(f"Balanced Acc:      {bal_acc:.4f}")
        if top3_acc is not None:
            print(f"Top-3 Accuracy:    {top3_acc:.4f}")

        results[name] = {"acc": acc, "f1_macro": f1_macro, "bal_acc": bal_acc, "top3_acc": top3_acc}

        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, X_test.columns, name)

    return results
