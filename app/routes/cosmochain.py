from flask import Blueprint, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

cosmo_bp = Blueprint('cosmochain', __name__, template_folder='../templates')

# Determine model directory under app/cosmochain_models/models
base_dir = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, '..', 'cosmochain_models', 'models')

# Load KMeans clustering models
kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
scaler_kmeans = joblib.load(os.path.join(model_dir, 'kmeans_scaler.pkl'))

# Load classification model and scaler
classification_model = joblib.load(os.path.join(model_dir, 'modele_classification_produit.pkl'))
scaler_class = joblib.load(os.path.join(model_dir, 'class_scaler.pkl'))
classifier = classification_model

# Prepare label encoder
label_encoder = LabelEncoder()
label_encoder.fit(["High Risk of Expiration", "Low Risk / Good Turnover"])

# Load regression model (and encoder if present)
regressor_tuple = joblib.load(os.path.join(model_dir, 'modele_prix_unitaire.pkl'))
try:
    regressor, encoder = regressor_tuple
except Exception:
    regressor = regressor_tuple
    encoder = None

@cosmo_bp.route('/')
def index():
    return render_template('./index.html')

@cosmo_bp.route('/predict_class', methods=['POST'])
def predict_class():
    try:
        data = request.form
        manufacture_date = pd.to_datetime(data["Manufacture_Date"])
        expiration_date = pd.to_datetime(data["Expiration_Date"])

        input_dict = {
            "rest_quantity": float(data["rest_quantity"]),
            "Quantite": float(data["Quantite"]),
            "Unit_Price": float(data["Unit_Price"]),
            "Manufacture_Date": manufacture_date.value,
            "Expiration_Date": expiration_date.value
        }
        X = pd.DataFrame([input_dict])
        # Align features to scaler
        try:
            expected = scaler_class.get_feature_names_out()
            for feat in expected:
                if feat not in X.columns:
                    X[feat] = 0
            X = X.loc[:, expected]
        except AttributeError:
            try:
                expected = classifier.feature_names_in_
                for feat in expected:
                    if feat not in X.columns:
                        X[feat] = 0
                X = X.loc[:, expected]
            except AttributeError:
                pass
        X_scaled = scaler_class.transform(X) if scaler_class is not None else X
        pred_index = classifier.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        return jsonify({"Product_Classification": pred_label})
    except Exception as e:
        import traceback
        return jsonify({"error": traceback.format_exc()}), 500

@cosmo_bp.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.form
        product = data["Product_Name"]
        quantite = float(data["Quantite"])
        category = data["Material_Category"]

        if encoder is not None:
            X_cat = encoder.transform([[product, category]])
        else:
            X_cat = pd.get_dummies(
                pd.DataFrame([[product, category]], columns=['Product_Name', 'Material_Category'])
            )
            try:
                all_cols = regressor.feature_names_in_.tolist()
                all_cols = [c for c in all_cols if c != 'Quantite']
            except AttributeError:
                all_cols = []
            for col in all_cols:
                if col not in X_cat.columns:
                    X_cat[col] = 0
            X_cat = X_cat[all_cols].values

        X_final = np.concatenate([X_cat, [[quantite]]], axis=1)
        price_pred = regressor.predict(X_final)[0]
        return jsonify({"Unit_Price_Prediction": round(price_pred, 2)})
    except Exception as e:
        import traceback
        return jsonify({"error": traceback.format_exc()}), 500

@cosmo_bp.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        data = request.get_json()
        features = ['rest_quantity', 'shelf_life', 'days_since_manufacture']
        if not all(f in data for f in features):
            return jsonify({'error': 'Missing required fields'}), 400
        df = pd.DataFrame([data])
        scaled = scaler_kmeans.transform(df[features])
        cluster_id = int(kmeans.predict(scaled)[0])
        labels = {
            0: "⚠️ High Overstock Risk",
            1: "✅ Balanced Inventory",
            2: "🔄 Rapid Turnover"
        }
        return jsonify({'cluster': cluster_id, 'label': labels.get(cluster_id)})
    except Exception as e:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500

@cosmo_bp.route('/predict_forecast', methods=['GET'])
def predict_forecast():
    try:
        st_model = joblib.load(os.path.join(model_dir, 'St_model.pkl'))
        future = st_model.make_future_dataframe(periods=6, freq='M')
        forecast = st_model.predict(future)
        preds = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)
        preds['ds'] = preds['ds'].dt.strftime('%Y-%m')
        return jsonify(preds.to_dict(orient='records'))
    except Exception as e:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500
