from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, top_k_accuracy_score, classification_report

def evaluate_predictions(model_name, y_test, predictions, probs, classes):
    acc = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    bal_acc = balanced_accuracy_score(y_test, predictions)
    
    top3_acc = None
    if probs is not None:
        try:
            top3_acc = top_k_accuracy_score(y_test, probs, k=3, labels=classes)
        except Exception:
            pass
            
    print(f"\n{model_name}")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Macro: {f1_macro:.2f}")
    print(f"Balanced Accuracy: {bal_acc:.2f}")
    if top3_acc is not None:
        print(f"Top-3 Accuracy: {top3_acc:.2f}")
        
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "bal_acc": bal_acc,
        "top3_acc": top3_acc
    }
