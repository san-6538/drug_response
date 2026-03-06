import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_importance(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        return
        
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:20] # Top 20 features
    
    sorted_importance = importance[idx]
    sorted_features = np.array(feature_names)[idx]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis", hue=sorted_features, legend=False)
    plt.title(f'{model_name} Feature Importance (Top 20)')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    filename = model_name.lower().replace(" ", "_") + "_importance.png"
    plt.savefig(os.path.join("results/plots", filename))
    plt.close()
