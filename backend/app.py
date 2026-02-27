"""
FastAPI Backend for CNN-LSTM Intrusion Detection System
Loads trained model (ids_model_package.h5 + .joblib)
Provides REST APIs for predictions, explanations, and health checks.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xai_explainer import XAIExplainer, check_xai_dependencies

# ----------------------------------------------------------------------
# TensorFlow Imports
# ----------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available — model loading disabled.")

# ----------------------------------------------------------------------
# Initialize FastAPI App
# ----------------------------------------------------------------------
app = FastAPI(title="CNN-LSTM Intrusion Detection System")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict later to specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load Model Package
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ids_model_package.joblib")
MODEL_H5_PATH = os.path.join(BASE_DIR, "..", "ids_model_package.h5")

print("📦 Loading model package...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ ids_model_package.joblib not found! Please train the model first.")

pkg = joblib.load(MODEL_PATH)
scaler = pkg["scaler"]
label_encoder = pkg["label_encoder"]
feature_names = pkg["feature_names"]

model = None
xai_explainer = None
if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_H5_PATH):
    print("🧠 Loading CNN-LSTM TensorFlow model...")
    model = load_model(MODEL_H5_PATH)
    
    # Initialize XAI explainer
    try:
        # Try to load sample training data to improve LIME explanations if available
        training_data = None
        try:
            training_csv = os.path.join(BASE_DIR, "..", "dataset", "generated_network_traffic.csv")
            if os.path.exists(training_csv):
                df_train = pd.read_csv(training_csv, low_memory=False)
                df_train.columns = df_train.columns.str.strip()
                # Keep only feature columns present in the model
                missing_cols = [c for c in feature_names if c not in df_train.columns]
                if not missing_cols:
                    training_data = df_train[feature_names].replace([np.inf, -np.inf], np.nan).dropna()
                    print(f"📚 Loaded training data for XAI ({len(training_data)} rows)")
                else:
                    print(f"⚠️ Training CSV missing columns for XAI: {missing_cols[:5]}")
        except Exception as e:
            print(f"⚠️ Failed to load training CSV for XAI: {e}")

        xai_explainer = XAIExplainer(model, scaler, feature_names, label_encoder, training_data=training_data)
        print("🔍 XAI explainer initialized")
    except Exception as e:
        print(f"⚠️ XAI explainer initialization failed: {e}")
        xai_explainer = None
else:
    print("⚠️ Model (.h5) not found or TensorFlow unavailable, cannot perform predictions.")

print("✅ Model and encoders loaded successfully.")
print(f"📊 Total features: {len(feature_names)}")
print(f"🏷️ Classes: {list(label_encoder.classes_)}")

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.get("/")
def root():
    """Simple root message."""
    return {"message": "CNN-LSTM Intrusion Detection Backend is running successfully."}


@app.get("/health")
def health_check():
    """Return model status and environment info."""
    xai_status = check_xai_dependencies()
    return {
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "model_loaded": model is not None,
        "xai_available": xai_explainer is not None,
        "num_features": len(feature_names),
        "classes": list(label_encoder.classes_),
        "xai_dependencies": xai_status,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CSV file to get predictions.
    Must contain the same feature columns used during training.
    """
    try:
        start = time.time()
        df = pd.read_csv(file.file, low_memory=False)
        df.columns = df.columns.str.strip()

        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols[:10]}")

        X = df[feature_names].replace([np.inf, -np.inf], np.nan).dropna()
        X_scaled = scaler.transform(X)

        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded on server.")

        preds = model.predict(X_scaled)
        y_pred = np.argmax(preds, axis=1)
        predicted_labels = label_encoder.inverse_transform(y_pred)
        df["Predicted_Label"] = predicted_labels

        latency = time.time() - start
        return {
            "rows_predicted": len(df),
            "predicted_classes": df["Predicted_Label"].value_counts().to_dict(),
            "processing_time_sec": round(latency, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict-single")
async def predict_single(data: dict):
    """
    Predict for a single record via JSON body.
    Example input:
    {
        "Flow Duration": 100,
        "Total Fwd Packets": 5,
        "Total Backward Packets": 3,
        ...
    }
    """
    try:
        X_input = pd.DataFrame([data])
        X_input = X_input[feature_names]
        X_scaled = scaler.transform(X_input)

        preds = model.predict(X_scaled)
        y_pred = np.argmax(preds, axis=1)
        label = label_encoder.inverse_transform(y_pred)[0]
        confidence = float(np.max(preds))

        return {
            "predicted_label": label,
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(label_encoder.classes_, preds[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/explain-prediction")
async def explain_prediction(data: dict):
    """
    Get XAI explanation for a single prediction.
    Includes LIME/SHAP explanations and risk assessment.
    """
    try:
        if xai_explainer is None:
            raise HTTPException(status_code=503, detail="XAI explainer not available")
        
        # Convert to DataFrame
        X_input = pd.DataFrame([data])
        X_input = X_input[feature_names]
        
        # Get basic prediction
        X_scaled = scaler.transform(X_input)
        preds = model.predict(X_scaled)
        y_pred = np.argmax(preds, axis=1)
        label = label_encoder.inverse_transform(y_pred)[0]
        confidence = float(np.max(preds))
        
        # Get XAI explanations
        explanations = {}
        
        # Try LIME explanation
        lime_result = xai_explainer.explain_prediction_lime(X_input, num_features=10)
        if "error" not in lime_result:
            explanations["lime"] = lime_result
        
        # Try SHAP explanation  
        shap_result = xai_explainer.explain_prediction_shap(X_input, max_evals=50)
        if "error" not in shap_result:
            explanations["shap"] = shap_result
        
        # Use the best available explanation for risk assessment
        best_explanation = explanations.get("lime", explanations.get("shap", {}))
        if not best_explanation:
            best_explanation = {
                "predicted_class": label,
                "confidence": confidence,
                "feature_importance": []
            }
        
        # Get risk assessment and alerts
        risk_assessment = xai_explainer.get_attack_risk_assessment(X_input, best_explanation)
        
        # Alerts are disabled in this deployment — do not create/send alerts
        # alert_created flag is always False when alert system is not present
        
        return {
            "prediction": {
                "predicted_label": label,
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(label_encoder.classes_, preds[0])
                }
            },
            "explanations": explanations,
            "risk_assessment": risk_assessment,
            "available_methods": list(explanations.keys()),
            "alert_created": label != 'BENIGN'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.post("/install-xai-dependencies")
async def install_xai_dependencies():
    """
    Install XAI dependencies (LIME and SHAP)
    """
    try:
        import subprocess
        import sys
        
        packages = ["lime", "shap"]
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        return {
            "message": "XAI dependencies installed successfully",
            "installed_packages": packages,
            "restart_required": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Installation failed: {e}")



# ----------------------------------------------------------------------
# SERVER ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server on http://127.0.0.1:8000 ...")
    os.system("cd e:\\ids\\backend")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
