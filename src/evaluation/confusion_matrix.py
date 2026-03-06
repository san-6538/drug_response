import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_cm(y_true, y_pred, model_name, target_le):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    filename = model_name.lower().replace(" ", "_") + "_cm.png"
    plt.savefig(os.path.join("results/plots", filename))
    plt.close()
