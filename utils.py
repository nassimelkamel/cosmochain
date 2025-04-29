# Utility functions for the Cosmetic Supply Chain Streamlit App

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
# import shap
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GRU
# from tensorflow.keras.optimizers import Adam
import io
import joblib
import os

# Define paths relative to this script's location
SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.joblib")
PCA_PATH = os.path.join(MODELS_DIR, "pca.joblib")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans.joblib")
DBSCAN_PATH = os.path.join(MODELS_DIR, "dbscan.joblib")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
GRU_MODEL_PATH = os.path.join(MODELS_DIR, "gru_model.h5")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(uploaded_file):
    """Loads data from an uploaded Excel file."""
    if uploaded_file is not None:
        try:
            # Determine the engine based on file extension
            if uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')
            else: # Assume .xlsx
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            print(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    return None

def preprocess_data(df, target_col=None, fit_preprocessors=False):
    """Cleans, encodes, and scales the data. Fits preprocessors if specified."""
    df_processed = df.copy()

    # Define columns based on notebook analysis
    numerical_cols = ['Dosage']
    categorical_cols = ['Product Name', 'Category', 'Brand Name', 'Material Name', 'Material Category']
    all_feature_cols = numerical_cols + categorical_cols

    # 1. Handle Missing Values (only for Dosage as per notebook)
    for col in numerical_cols:
        if col in df_processed.columns:
            if fit_preprocessors:
                mean_val = df_processed[col].mean()
                # Save mean value if needed, or handle within scaler
            # For simplicity, just fillna here. Scaler handles means during fit.
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        else:
            print(f"⚠️ Warning: Numerical column '{col}' not found. Skipping missing value imputation.")

    # 2. Encode Categorical Columns
    label_encoders = {}
    if fit_preprocessors:
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                # Ensure column is string type before encoding
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
            else:
                print(f"⚠️ Warning: Categorical column '{col}' not found. Skipping encoding.")
        # Save fitted label encoders
        joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
        print(f"✅ Label encoders fitted and saved to {LABEL_ENCODERS_PATH}")
    else:
        # Load fitted label encoders
        if os.path.exists(LABEL_ENCODERS_PATH):
            label_encoders = joblib.load(LABEL_ENCODERS_PATH)
            for col in categorical_cols:
                if col in df_processed.columns and col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen labels during transform - map to a default value (e.g., -1 or len(classes))
                    # Convert to string first
                    df_processed[col] = df_processed[col].astype(str)
                    known_classes = list(le.classes_)
                    df_processed[col] = df_processed[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1) # Assign -1 for unknown
                    # Alternative: Fit on the fly (might not be ideal for consistency)
                    # df_processed[col] = le.transform(df_processed[col].astype(str))
                elif col in df_processed.columns:
                     print(f"⚠️ Warning: Label encoder for column '{col}' not found. Skipping encoding.")
                else:
                     print(f"⚠️ Warning: Categorical column '{col}' not found. Skipping encoding.")
        else:
            print(f"⚠️ Error: Fitted label encoders not found at {LABEL_ENCODERS_PATH}. Cannot encode data.")
            return None, None # Indicate failure

    print("✅ Data cleaned and encoded.")

    # Separate features (X) and target (y) if target_col is provided
    X = df_processed.copy()
    y = None
    if target_col:
        if target_col in X.columns:
            y = X.pop(target_col)
            # Ensure target is also label encoded if it's categorical and not already done
            if target_col in categorical_cols and not fit_preprocessors and os.path.exists(LABEL_ENCODERS_PATH):
                 if target_col in label_encoders:
                     le_target = label_encoders[target_col]
                     known_classes_target = list(le_target.classes_)
                     # Apply transformation to the original target column before popping if needed
                     # This assumes y was popped *after* encoding, which might be wrong.
                     # Let's re-encode y separately if needed.
                     y = df[target_col].astype(str) # Get original target
                     y = y.apply(lambda x: le_target.transform([x])[0] if x in known_classes_target else -1)
                 else:
                     print(f"⚠️ Warning: Label encoder for target column '{target_col}' not found.")
            elif target_col in categorical_cols and fit_preprocessors:
                 # y should already be encoded from the loop above
                 pass
        else:
            print(f"⚠️ Warning: Target column '{target_col}' not found in DataFrame.")
            target_col = None # Proceed without target

    # Select only feature columns that exist in the dataframe
    feature_cols_present = [col for col in all_feature_cols if col in X.columns and col != target_col]
    X = X[feature_cols_present]

    # 3. Feature Scaling (only on feature columns)
    if fit_preprocessors:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Save fitted scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler fitted and saved to {SCALER_PATH}")
    else:
        # Load fitted scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            X_scaled = scaler.transform(X)
        else:
            print(f"⚠️ Error: Fitted scaler not found at {SCALER_PATH}. Cannot scale data.")
            return None, None # Indicate failure

    print("✅ Features scaled.")
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled_df, y

# --- Placeholder functions for models --- 

def train_save_cluster_models(X_scaled_df):
    """Trains and saves PCA, KMeans, and DBSCAN models."""
    print("Training clustering models...")
    try:
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled_df)
        joblib.dump(pca, PCA_PATH)
        print(f"✅ PCA fitted and saved to {PCA_PATH}")

        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Specify n_init
        kmeans.fit(X_pca)
        joblib.dump(kmeans, KMEANS_PATH)
        print(f"✅ KMeans fitted and saved to {KMEANS_PATH}")

        # DBSCAN (Parameters might need tuning based on data)
        dbscan = DBSCAN(eps=0.5, min_samples=5) # Example parameters
        dbscan.fit(X_scaled_df) # DBSCAN often works better on original scaled data
        joblib.dump(dbscan, DBSCAN_PATH)
        print(f"✅ DBSCAN fitted and saved to {DBSCAN_PATH}")
        print("✅ Clustering models trained and saved.")
        return True
    except Exception as e:
        print(f"Error training/saving clustering models: {e}")
        return False

def run_clustering(X_scaled_df):
    """Loads and runs pre-trained clustering models."""
    results = {}
    try:
        # Load PCA
        if os.path.exists(PCA_PATH):
            pca = joblib.load(PCA_PATH)
            X_pca = pca.transform(X_scaled_df)
            results['pca_components'] = X_pca
        else:
            print(f"⚠️ PCA model not found at {PCA_PATH}")
            return None

        # Load and run KMeans
        if os.path.exists(KMEANS_PATH):
            kmeans = joblib.load(KMEANS_PATH)
            kmeans_labels = kmeans.predict(X_pca)
            results['kmeans_labels'] = kmeans_labels
        else:
            print(f"⚠️ KMeans model not found at {KMEANS_PATH}")

        # Load and run DBSCAN
        if os.path.exists(DBSCAN_PATH):
            dbscan = joblib.load(DBSCAN_PATH)
            dbscan_labels = dbscan.fit_predict(X_scaled_df) # fit_predict for DBSCAN
            results['dbscan_labels'] = dbscan_labels
        else:
            print(f"⚠️ DBSCAN model not found at {DBSCAN_PATH}")

        print("✅ Clustering predictions generated.")
        return results

    except Exception as e:
        print(f"Error running clustering models: {e}")
        return None

# Add placeholders for other models (LGBM, RF, GRU) - train/load and predict

print("Utils module loaded.")





def train_save_lgbm_model(X_train, y_train):
    """Trains and saves the LightGBM classification model."""
    print("Training LightGBM model...")
    try:
        # Parameters from notebook (or default)
        lgbm_clf = lgb.LGBMClassifier(random_state=42)
        lgbm_clf.fit(X_train, y_train)
        joblib.dump(lgbm_clf, LGBM_MODEL_PATH)
        print(f"✅ LightGBM model trained and saved to {LGBM_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error training/saving LightGBM model: {e}")
        return False

def run_lgbm_prediction(X_test):
    """Loads and runs the pre-trained LightGBM model for prediction."""
    try:
        if os.path.exists(LGBM_MODEL_PATH):
            lgbm_clf = joblib.load(LGBM_MODEL_PATH)
            y_pred = lgbm_clf.predict(X_test)
            y_proba = lgbm_clf.predict_proba(X_test)
            print("✅ LightGBM predictions generated.")
            # SHAP values (optional, requires trained model and data)
            # explainer = shap.TreeExplainer(lgbm_clf)
            # shap_values = explainer.shap_values(X_test)
            return y_pred, y_proba #, shap_values
        else:
            print(f"⚠️ LightGBM model not found at {LGBM_MODEL_PATH}")
            return None, None
    except Exception as e:
        print(f"Error running LightGBM prediction: {e}")
        return None, None

def train_save_rf_model(X_train, y_train):
    """Trains and saves the RandomForest classification model using best params from notebook."""
    print("Training RandomForest model...")
    try:
        # Using best parameters found in the notebook's GridSearchCV
        # Note: GridSearchCV itself is not run here for speed, using its result.
        # best_params_ = {
        #     'max_depth': 10, # Example, adjust based on notebook output
        #     'min_samples_leaf': 1, # Example
        #     'min_samples_split': 2, # Example
        #     'n_estimators': 100 # Example
        # }
        # rf_clf = RandomForestClassifier(random_state=42, **best_params_)
        # Simpler approach: Use default or notebook's initial parameters if grid search is too complex to replicate
        rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1) # From notebook example

        rf_clf.fit(X_train, y_train)
        joblib.dump(rf_clf, RF_MODEL_PATH)
        print(f"✅ RandomForest model trained and saved to {RF_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error training/saving RandomForest model: {e}")
        return False

def run_rf_prediction(X_test):
    """Loads and runs the pre-trained RandomForest model for prediction."""
    try:
        if os.path.exists(RF_MODEL_PATH):
            rf_clf = joblib.load(RF_MODEL_PATH)
            y_pred = rf_clf.predict(X_test)
            y_proba = rf_clf.predict_proba(X_test)
            importances = rf_clf.feature_importances_
            print("✅ RandomForest predictions and feature importances generated.")
            return y_pred, y_proba, importances
        else:
            print(f"⚠️ RandomForest model not found at {RF_MODEL_PATH}")
            return None, None, None
    except Exception as e:
        print(f"Error running RandomForest prediction: {e}")
        return None, None, None

# --- GRU Model Functions ---
# Note: GRU requires data in sequences (samples, timesteps, features).
# The preprocessing needs adjustment for GRU.
# Assuming classification task for now, adapting GRU might be complex.

def build_gru_model(input_shape, num_classes):
    """Builds a simple GRU model for classification."""
    model = Sequential([
        GRU(50, input_shape=input_shape, return_sequences=False), # Adjust units/layers as needed
        Dense(num_classes, activation='softmax') # Use 'sigmoid' for binary
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Use 'binary_crossentropy' for binary
    return model

def prepare_data_for_gru(X, y, timesteps=1):
    """Reshapes data into sequences for GRU. Placeholder implementation."""
    # This needs careful implementation based on the actual time-series nature of the data.
    # For a simple example, we treat each row as a sequence of length 1.
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X
    num_samples = X_np.shape[0]
    num_features = X_np.shape[1]
    X_reshaped = X_np.reshape(num_samples, timesteps, num_features)

    # Ensure y is numpy array
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y_np = y.values.ravel() # Flatten if needed
    else:
        y_np = y

    return X_reshaped, y_np

def train_save_gru_model(X_train, y_train, timesteps=1, epochs=10, batch_size=32):
    """Trains and saves the GRU classification model."""
    print("Training GRU model...")
    try:
        X_train_gru, y_train_gru = prepare_data_for_gru(X_train, y_train, timesteps)
        if X_train_gru is None:
            return False

        input_shape = (X_train_gru.shape[1], X_train_gru.shape[2])
        num_classes = len(np.unique(y_train_gru[y_train_gru != -1])) # Exclude potential -1 from unknown labels
        if num_classes <= 1:
             print(f"⚠️ Error: Not enough classes ({num_classes}) found in target variable for GRU training.")
             return False

        gru_model = build_gru_model(input_shape, num_classes)
        print(gru_model.summary())

        # Filter out samples with unknown target label (-1)
        valid_indices = y_train_gru != -1
        if not np.any(valid_indices):
            print("⚠️ Error: No valid target labels found for GRU training.")
            return False

        X_train_gru_valid = X_train_gru[valid_indices]
        y_train_gru_valid = y_train_gru[valid_indices]

        history = gru_model.fit(X_train_gru_valid, y_train_gru_valid, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

        gru_model.save(GRU_MODEL_PATH)
        print(f"✅ GRU model trained and saved to {GRU_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error training/saving GRU model: {e}")
        return False

def run_gru_prediction(X_test, timesteps=1):
    """Loads and runs the pre-trained GRU model for prediction."""
    from tensorflow.keras.models import load_model
    try:
        if os.path.exists(GRU_MODEL_PATH):
            gru_model = load_model(GRU_MODEL_PATH)
            X_test_gru, _ = prepare_data_for_gru(X_test, None, timesteps) # y is not needed for prediction
            if X_test_gru is None:
                return None, None

            y_proba = gru_model.predict(X_test_gru)
            y_pred = np.argmax(y_proba, axis=1)
            print("✅ GRU predictions generated.")
            return y_pred, y_proba
        else:
            print(f"⚠️ GRU model not found at {GRU_MODEL_PATH}")
            return None, None
    except Exception as e:
        print(f"Error running GRU prediction: {e}")
        return None, None

print("Utils module updated with classification and GRU models.")


