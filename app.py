# Streamlit App for Cosmetic Supply Chain ML Pipeline - Futuristic Redesign v2

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Import utility functions
import utils

# Define paths relative to this script's location
SCRIPT_DIR_APP = os.path.dirname(__file__)
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR_APP, "data", "cosmetic_products_base_materials2.xlsx")
STATIC_PREVIEWS_DIR = os.path.join(SCRIPT_DIR_APP, "static_previews")
KMEANS_PREVIEW_PATH = os.path.join(STATIC_PREVIEWS_DIR, "kmeans_pca_preview.png")
RF_IMPORTANCE_PREVIEW_PATH = os.path.join(STATIC_PREVIEWS_DIR, "rf_importance_preview.png")

# --- Page Configuration (Wide layout, Initial Sidebar Expanded) ---
st.set_page_config(layout="wide", page_title="Cosmetic AI Pipeline", initial_sidebar_state="expanded")

# --- Custom CSS for Futuristic Dark Theme ---
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #0E1117; /* Streamlit dark theme background */
        color: #FAFAFA; /* Light text */
    }

    /* Sidebar */
    .stSidebar {
        background-color: #1A1C24; /* Slightly lighter dark */
    }
    .stSidebar .stMarkdown, .stSidebar .stButton > button, .stSidebar .stSelectbox > label {
        color: #E1E1E1; /* Lighter text for sidebar */
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #00C4FF; /* Bright blue for headers */
    }

    /* Buttons */
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stButton > button:active {
        background-color: #003d80;
    }

    /* Expanders */
    .stExpander {
        border: 1px solid #444;
        border-radius: 8px;
        background-color: #1A1C24;
    }
    .stExpander header {
        font-weight: bold;
        color: #00C4FF;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A1C24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px;
        color: #E1E1E1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0E1117; /* Match app background */
        color: #00C4FF; /* Highlight selected tab */
        font-weight: bold;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid #444;
        border-radius: 8px;
    }

    /* Metrics */
    .stMetric {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00C4FF;
    }

</style>
""", unsafe_allow_html=True)

st.title("🚀 Cosmetic AI Supply Chain Pipeline")

st.sidebar.title("🌌 Navigation & Controls")

# --- Global Variables/State Management ---
if "df" not in st.session_state:
    st.session_state.df = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "y" not in st.session_state:
    st.session_state.y = None
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False # Track if models have been trained in this session

# --- Load and Preprocess Data (AUTOMATIC) ---
@st.cache_data # Cache the loaded raw data
def load_raw_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Default data file not found at {path}")
        return None
    try:
        if path.endswith(".xls"):
            df = pd.read_excel(path, engine='xlrd')
        else:
            df = pd.read_excel(path, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading data from {path}: {e}")
        return None

@st.cache_resource # Cache preprocessed data and fitted objects
def preprocess_data_cached(df, fit_preprocessors=False):
    if df is None:
        return None, None, None, None
    target_col = "Category"
    try:
        X_scaled_df, y_series = utils.preprocess_data(df.copy(), target_col=target_col, fit_preprocessors=fit_preprocessors)
        label_encoders = None
        if os.path.exists(utils.LABEL_ENCODERS_PATH):
             label_encoders = joblib.load(utils.LABEL_ENCODERS_PATH)
        feature_names = X_scaled_df.columns.tolist() if X_scaled_df is not None else None
        return X_scaled_df, y_series, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None, None, None, None

# --- Check if models exist (basic check) ---
@st.cache_resource
def check_models_exist():
    paths = [utils.SCALER_PATH, utils.LABEL_ENCODERS_PATH, utils.PCA_PATH, 
             utils.KMEANS_PATH, utils.DBSCAN_PATH, utils.LGBM_MODEL_PATH, 
             utils.RF_MODEL_PATH, utils.GRU_MODEL_PATH]
    # Check if the directory exists first
    if not os.path.exists(utils.MODELS_DIR):
        return False
    return all(os.path.exists(p) for p in paths)

# --- Load data automatically on startup ---
if not st.session_state.data_loaded:
    with st.spinner("🛰️ Loading and preprocessing cosmic data..."):
        st.session_state.df = load_raw_data(DEFAULT_DATA_PATH)
        if st.session_state.df is not None:
            # Preprocess using existing fitted objects if models were trained before
            fit = not check_models_exist()
            X_scaled_df, y_series, label_encoders, feature_names = preprocess_data_cached(st.session_state.df, fit_preprocessors=fit)
            if X_scaled_df is not None:
                st.session_state.X_scaled = X_scaled_df
                st.session_state.y = y_series
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_names = feature_names
                st.session_state.data_loaded = True
                st.sidebar.success("Data loaded & preprocessed.")
            else:
                st.sidebar.error("Preprocessing failed.")
        else:
            st.sidebar.error("Data loading failed.")

# Update model trained status
if not st.session_state.models_trained:
    st.session_state.models_trained = check_models_exist()

# --- Sidebar: Model Training Control ---
st.sidebar.header("🤖 Model Training")
if st.session_state.data_loaded:
    if st.session_state.models_trained:
        st.sidebar.success("Models are trained and ready! ✨")
    else:
        st.sidebar.warning("Models need training.")

    if st.sidebar.button("Train / Retrain All Models", key="train_button"):
        with st.spinner("🚀 Initiating training sequence..."):
            # Ensure data is loaded and preprocessed with fit_preprocessors=True
            with st.spinner("Preprocessing data for training..."):
                # Clear cache for preprocessing to ensure it runs with fit=True
                preprocess_data_cached.clear()
                X_scaled_df, y_series, _, _ = preprocess_data_cached(st.session_state.df, fit_preprocessors=True)
            
            if X_scaled_df is not None and y_series is not None:
                st.session_state.X_scaled = X_scaled_df # Update state with newly fitted preprocessors
                st.session_state.y = y_series
                st.session_state.label_encoders = joblib.load(utils.LABEL_ENCODERS_PATH) # Reload fitted encoders
                st.session_state.feature_names = X_scaled_df.columns.tolist()
                st.success("Data preprocessed for training.")
                
                # Filter out invalid target labels (-1) before splitting
                valid_indices_train = st.session_state.y != -1
                X_scaled_valid = st.session_state.X_scaled[valid_indices_train]
                y_valid = st.session_state.y[valid_indices_train]

                if len(y_valid) == 0:
                    st.error("No valid target labels found. Cannot train classification models.")
                else:
                    X_train, X_test, y_train, y_test = utils.train_test_split(X_scaled_valid, y_valid, test_size=0.3, random_state=42, stratify=y_valid)
                    
                    training_successful = True
                    model_status = {}

                    # Train Clustering
                    with st.spinner("Training Clustering models..."):
                        if not utils.train_save_cluster_models(X_scaled_valid):
                            model_status["Clustering"] = "❌ Failed"
                            training_successful = False
                        else:
                            model_status["Clustering"] = "✅ Success"
                    
                    # Train LightGBM
                    with st.spinner("Training LightGBM..."):
                        if not utils.train_save_lgbm_model(X_train, y_train):
                            model_status["LightGBM"] = "❌ Failed"
                            training_successful = False
                        else:
                            model_status["LightGBM"] = "✅ Success"

                    # Train RandomForest
                    with st.spinner("Training RandomForest..."):
                        if not utils.train_save_rf_model(X_train, y_train):
                            model_status["RandomForest"] = "❌ Failed"
                            training_successful = False
                        else:
                            model_status["RandomForest"] = "✅ Success"

                    # Train GRU
                    with st.spinner("Training GRU..."):
                        if not utils.train_save_gru_model(X_train, y_train, timesteps=1, epochs=10):
                            model_status["GRU"] = "❌ Failed"
                            training_successful = False
                        else:
                            model_status["GRU"] = "✅ Success"
                    
                    # Display status
                    st.sidebar.subheader("Training Status:")
                    for model, status in model_status.items():
                        st.sidebar.write(f"- {model}: {status}")

                    if training_successful:
                        st.session_state.models_trained = True
                        st.balloons()
                        st.sidebar.success("All models trained successfully!")
                        # Clear cache for preprocessing again, so next load uses saved models
                        preprocess_data_cached.clear()
                    else:
                        st.sidebar.error("Some models failed to train.")
            else:
                st.error("Cannot train models. Data loading/preprocessing failed.")
else:
    st.sidebar.error("Data not loaded. Cannot train models.")

# --- Main Application Area --- 

if not st.session_state.data_loaded:
    st.error("Pipeline initiation failed: Could not load or process the core dataset. Check logs or data file path.")
else:
    # --- Tabs for Different Sections ---
    tab_intro, tab_data, tab_cluster, tab_classify, tab_deep = st.tabs([
        "🌌 Introduction", 
        "📊 Data Explorer", 
        "✨ Clustering Insights", 
        "🎯 Classification Center", 
        "🧠 Deep Learning (GRU)"
    ])

    # --- Introduction Tab ---
    with tab_intro:
        st.header("Welcome to the Cosmetic AI Pipeline")
        st.markdown("""
        This dashboard provides a futuristic interface to interact with machine learning models applied to cosmetic supply chain data. 
        Navigate through the tabs to explore data insights, clustering results, classification predictions, and deep learning model performance.

        **Key Features:**
        *   **Automatic Data Handling:** The core dataset is loaded and preprocessed automatically.
        *   **Static Previews:** See example visualizations before training models.
        *   **Interactive Visualizations:** Explore model results through dynamic charts and tables.
        *   **On-Demand Training:** Train or retrain all models directly from the sidebar.
        *   **Modern Interface:** Utilizing tabs, expanders, and custom styling for a clean look.
        
        *Ensure models are trained using the sidebar button if this is the first run or if you need to update them.*
        """)
        if not st.session_state.models_trained:
            st.warning("Warning: Models have not been trained yet. Please use the \"Train / Retrain All Models\" button in the sidebar.")

    # --- Data Explorer Tab ---
    with tab_data:
        st.header("📊 Data Explorer")
        st.markdown("Explore the raw and preprocessed cosmetic supply chain data.")
        
        with st.expander("Raw Data Sample", expanded=False):
            st.dataframe(st.session_state.df.head(10))
        
        with st.expander("Data Statistics", expanded=False):
            st.dataframe(st.session_state.df.describe(include='all'))
            
        if st.session_state.X_scaled is not None:
            with st.expander("Preprocessed Data (Features - Scaled & Encoded)", expanded=False):
                st.dataframe(st.session_state.X_scaled.head(10))
        else:
            st.warning("Preprocessed feature data not available.")
            
        if st.session_state.y is not None:
             with st.expander("Target Variable (Encoded)", expanded=False):
                st.dataframe(st.session_state.y.head(10))
        else:
            st.warning("Target variable data not available.")

    # --- Clustering Insights Tab ---
    with tab_cluster:
        st.header("✨ Clustering Insights")
        if not st.session_state.models_trained:
            st.warning("Models need to be trained first. Use the sidebar button.")
            st.subheader("Example Visualization (KMeans PCA)")
            if os.path.exists(KMEANS_PREVIEW_PATH):
                st.image(KMEANS_PREVIEW_PATH, caption="Example KMeans Clustering on PCA Components")
            else:
                st.info("Static preview image not found.")
        elif st.session_state.X_scaled is None:
            st.error("Preprocessed data not available for clustering.")
        else:
            st.info("Analyzing clusters based on product/material features...")
            cluster_results = utils.run_clustering(st.session_state.X_scaled)

            if cluster_results:
                df_display_cluster = st.session_state.df.copy()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("KMeans Results")
                    if "kmeans_labels" in cluster_results and "pca_components" in cluster_results:
                        df_display_cluster["KMeans_Cluster"] = cluster_results["kmeans_labels"]
                        fig_kmeans, ax_kmeans = plt.subplots(figsize=(7, 6))
                        scatter = ax_kmeans.scatter(cluster_results["pca_components"][:, 0], 
                                                    cluster_results["pca_components"][:, 1], 
                                                    c=cluster_results["kmeans_labels"], cmap='viridis'
                                                    )
                        ax_kmeans.set_title("KMeans Clustering (PCA)")
                        ax_kmeans.set_xlabel("PCA Component 1")
                        ax_kmeans.set_ylabel("PCA Component 2")
                        legend1 = ax_kmeans.legend(*scatter.legend_elements(), title="Clusters")
                        ax_kmeans.add_artist(legend1)
                        fig_kmeans.patch.set_alpha(0) # Make plot background transparent
                        ax_kmeans.set_facecolor("#0E1117") # Match app background
                        ax_kmeans.tick_params(colors='#FAFAFA', which='both')
                        ax_kmeans.xaxis.label.set_color('#FAFAFA')
                        ax_kmeans.yaxis.label.set_color('#FAFAFA')
                        ax_kmeans.title.set_color('#00C4FF')
                        if legend1:
                            plt.setp(legend1.get_texts(), color='#FAFAFA')
                            legend1.get_title().set_color('#FAFAFA')
                        st.pyplot(fig_kmeans)
                        plt.close(fig_kmeans)
                    else:
                        st.warning("KMeans results or PCA components not available.")

                with col2:
                    st.subheader("DBSCAN Results")
                    if "dbscan_labels" in cluster_results:
                        df_display_cluster["DBSCAN_Cluster"] = cluster_results["dbscan_labels"]
                        st.write("Cluster Distribution:")
                        # Use st.dataframe for better styling
                        st.dataframe(df_display_cluster["DBSCAN_Cluster"].value_counts().reset_index().rename(columns={"index": "Cluster", "DBSCAN_Cluster": "Count"}))
                    else:
                        st.warning("DBSCAN results not available.")

                with st.expander("Data with Cluster Labels", expanded=False):
                    st.dataframe(df_display_cluster.head(20))
            else:
                st.error("Failed to run clustering models.")

    # --- Classification Center Tab ---
    with tab_classify:
        st.header("🎯 Classification Center")
        if not st.session_state.models_trained:
            st.warning("Models need to be trained first. Use the sidebar button.")
            st.subheader("Example Visualization (RandomForest Feature Importance)")
            if os.path.exists(RF_IMPORTANCE_PREVIEW_PATH):
                st.image(RF_IMPORTANCE_PREVIEW_PATH, caption="Example RandomForest Feature Importance")
            else:
                st.info("Static preview image not found.")
        elif st.session_state.X_scaled is None or st.session_state.y is None:
            st.error("Preprocessed data or target variable not available for classification.")
        else:
            st.info("Predicting product categories using LightGBM and RandomForest...")
            X_test_data = st.session_state.X_scaled
            y_true = st.session_state.y
            target_col_name = "Category"
            le_target = st.session_state.label_encoders.get(target_col_name) if st.session_state.label_encoders else None

            # --- Function to display classification results ---
            def display_classification_results(model_name, y_pred, y_proba, feature_importances=None):
                st.subheader(f"{model_name} Performance")
                if y_pred is None:
                    st.error(f"Failed to get {model_name} predictions.")
                    return

                valid_indices = y_true != -1
                y_true_valid = y_true[valid_indices]
                y_pred_valid = y_pred[valid_indices]

                if len(y_true_valid) == 0:
                    st.warning(f"No valid true labels found to generate report for {model_name}.")
                    return

                report_str = "Could not generate report."
                cm_fig = None
                try:
                    if le_target:
                        y_true_decoded = le_target.inverse_transform(y_true_valid)
                        y_pred_decoded = le_target.inverse_transform(y_pred_valid)
                        report_str = utils.classification_report(y_true_decoded, y_pred_decoded, output_dict=False, zero_division=0)
                        cm = utils.confusion_matrix(y_true_decoded, y_pred_decoded, labels=le_target.classes_)
                        
                        cm_fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d',
                                    xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax, cmap="Blues")
                        ax.set_title(f"{model_name} Confusion Matrix")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        cm_fig.patch.set_alpha(0)
                        ax.set_facecolor("#0E1117")
                        ax.tick_params(colors='#FAFAFA', which='both')
                        ax.xaxis.label.set_color('#FAFAFA')
                        ax.yaxis.label.set_color('#FAFAFA')
                        ax.title.set_color('#00C4FF')
                        plt.setp(ax.get_xticklabels(), color='#FAFAFA')
                        plt.setp(ax.get_yticklabels(), color='#FAFAFA')
                    else:
                        report_str = utils.classification_report(y_true_valid, y_pred_valid, output_dict=False, zero_division=0)
                        st.warning("Label encoder not found. Displaying encoded results.")
                except Exception as e:
                    st.error(f"Error generating report/matrix for {model_name}: {e}")
                    report_str = f"Error: {e}"

                col1_cls, col2_cls = st.columns([1, 1])
                with col1_cls:
                    st.text("Classification Report:")
                    st.code(report_str, language=None)
                with col2_cls:
                    if cm_fig:
                        st.pyplot(cm_fig)
                        plt.close(cm_fig)
                    else:
                        st.write("Confusion matrix could not be generated.")
                
                # Feature Importances (if available)
                # Made expander open by default for RF
                expander_state = True if model_name == "RandomForest" else False
                if feature_importances is not None and st.session_state.feature_names:
                    with st.expander(f"{model_name} Feature Importances", expanded=expander_state):
                        try:
                            feature_imp = pd.Series(feature_importances, index=st.session_state.feature_names).sort_values(ascending=False)
                            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                            # Use the corrected palette call
                            sns.barplot(x=feature_imp.head(15), y=feature_imp.head(15).index, ax=ax_imp, palette="viridis")
                            ax_imp.set_title(f"{model_name} Top 15 Feature Importances")
                            ax_imp.set_xlabel("Importance Score")
                            ax_imp.set_ylabel("Features")
                            fig_imp.patch.set_alpha(0)
                            ax_imp.set_facecolor("#0E1117")
                            ax_imp.tick_params(colors='#FAFAFA', which='both')
                            ax_imp.xaxis.label.set_color('#FAFAFA')
                            ax_imp.yaxis.label.set_color('#FAFAFA')
                            ax_imp.title.set_color('#00C4FF')
                            plt.setp(ax_imp.get_xticklabels(), color='#FAFAFA')
                            plt.setp(ax_imp.get_yticklabels(), color='#FAFAFA')
                            st.pyplot(fig_imp)
                            plt.close(fig_imp)
                        except Exception as imp_err:
                            st.error(f"Error displaying feature importances: {imp_err}")
            
            # --- Run and Display --- 
            y_pred_lgbm, y_proba_lgbm = utils.run_lgbm_prediction(X_test_data)
            display_classification_results("LightGBM", y_pred_lgbm, y_proba_lgbm)
            
            st.divider()
            
            y_pred_rf, y_proba_rf, importances_rf = utils.run_rf_prediction(X_test_data)
            display_classification_results("RandomForest", y_pred_rf, y_proba_rf, importances_rf)

    # --- Deep Learning Tab ---
    with tab_deep:
        st.header("🧠 Deep Learning (GRU)")
        st.warning("Note: GRU model applied assuming each data row is a sequence step. Performance depends on data suitability for sequential modeling.")
        if not st.session_state.models_trained:
            st.warning("Models need to be trained first. Use the sidebar button.")
            # Add GRU preview if available/relevant
            # st.image(GRU_PREVIEW_PATH, caption="Example GRU Performance") 
        elif st.session_state.X_scaled is None or st.session_state.y is None:
            st.error("Preprocessed data or target variable not available for GRU.")
        else:
            st.info("Running GRU model prediction...")
            X_test_data = st.session_state.X_scaled
            y_true = st.session_state.y
            timesteps = 1 # As defined in utils
            le_target = st.session_state.label_encoders.get("Category") if st.session_state.label_encoders else None

            y_pred_gru, y_proba_gru = utils.run_gru_prediction(X_test_data, timesteps=timesteps)
            display_classification_results("GRU", y_pred_gru, y_proba_gru) # Reuse the display function

print("Streamlit app script updated with static previews and RF enhancement.")

