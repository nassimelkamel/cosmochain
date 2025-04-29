# Script to generate static preview images for Streamlit app

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Define paths relative to this script's location
SCRIPT_DIR = os.path.dirname(__file__)
STATIC_PREVIEWS_DIR = os.path.join(SCRIPT_DIR, "static_previews")
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "cosmetic_products_base_materials2.xlsx")

# Ensure static previews directory exists
os.makedirs(STATIC_PREVIEWS_DIR, exist_ok=True)

# --- Generate KMeans PCA Plot Preview ---
def generate_kmeans_preview():
    print("Generating KMeans PCA preview...")
    try:
        # Create dummy PCA data and labels
        np.random.seed(42)
        dummy_pca_data = np.random.rand(100, 2) * 10
        dummy_labels = np.random.randint(0, 3, 100)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(dummy_pca_data[:, 0], dummy_pca_data[:, 1], c=dummy_labels, cmap="viridis")
        ax.set_title("Example: KMeans Clustering (PCA)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        # Apply dark theme styling
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="#FAFAFA", which="both")
        ax.xaxis.label.set_color("#FAFAFA")
        ax.yaxis.label.set_color("#FAFAFA")
        ax.title.set_color("#00C4FF")
        if legend1:
            plt.setp(legend1.get_texts(), color="#FAFAFA")
            legend1.get_title().set_color("#FAFAFA")

        save_path = os.path.join(STATIC_PREVIEWS_DIR, "kmeans_pca_preview.png")
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved KMeans preview to {save_path}")
        return True
    except Exception as e:
        print(f"Error generating KMeans preview: {e}")
        return False

# --- Generate RandomForest Feature Importance Preview ---
def generate_rf_importance_preview():
    print("Generating RF Feature Importance preview...")
    try:
        # Create dummy feature importances
        dummy_features = [f"Feature_{i}" for i in range(1, 16)]
        dummy_importances = np.random.rand(15) * 100
        dummy_importances = np.sort(dummy_importances)[::-1] # Sort descending
        feature_imp = pd.Series(dummy_importances, index=dummy_features)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="viridis")
        ax.set_title("Example: RandomForest Top 15 Feature Importances")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")

        # Apply dark theme styling
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="#FAFAFA", which="both")
        ax.xaxis.label.set_color("#FAFAFA")
        ax.yaxis.label.set_color("#FAFAFA")
        ax.title.set_color("#00C4FF")
        plt.setp(ax.get_xticklabels(), color="#FAFAFA")
        plt.setp(ax.get_yticklabels(), color="#FAFAFA")

        save_path = os.path.join(STATIC_PREVIEWS_DIR, "rf_importance_preview.png")
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved RF Importance preview to {save_path}")
        return True
    except Exception as e:
        print(f"Error generating RF Importance preview: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting static preview generation...")
    kmeans_ok = generate_kmeans_preview()
    rf_ok = generate_rf_importance_preview()

    if kmeans_ok and rf_ok:
        print("Successfully generated all static previews.")
    else:
        print("Failed to generate some static previews.")

