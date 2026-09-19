from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Rohit's Fraud Detection Engine", version="2.0.0")

models = {}
scaler = None

# Adding static training metrics to enrich the API response
MODEL_METRICS = {
    "xgboost": {"accuracy": "99.96%", "precision": "94.5%", "recall": "85.2%", "f1_score": "89.6%"},
    "random_forest": {"accuracy": "99.95%", "precision": "92.1%", "recall": "78.4%", "f1_score": "84.7%"},
    "logistic_regression": {"accuracy": "97.46%", "precision": "6.5%", "recall": "91.0%", "f1_score": "12.2%"} # High recall, terrible precision baseline
}

class TransactionRequest(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.on_event("startup")
def load_assets():
    global models, scaler
    try:
        models['xgboost'] = joblib.load(BASE_DIR / "fraud_model.pkl")
        models['logistic_regression'] = joblib.load(BASE_DIR / "lr_model.pkl")
        models['random_forest'] = joblib.load(BASE_DIR / "rf_model.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        print("✅ All models and scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load assets: {e}")

@app.get("/")
def read_root():
    return FileResponse("index.html")

def get_risk_level(probability: float):
    if probability < 0.20: return "LOW"
    elif probability < 0.50: return "MEDIUM"
    elif probability < 0.80: return "HIGH"
    else: return "CRITICAL"

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest, model_type: str = Query("xgboost")):
    if not models or not scaler:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    try:
        df = pd.DataFrame([transaction.model_dump()])
        df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        feature_names = models['xgboost'].get_booster().feature_names
        df = df[feature_names]
        
        if model_type == "all":
            results = {}
            for name, m in models.items():
                pred = int(m.predict(df)[0])
                prob = float(m.predict_proba(df)[0][1])
                results[name] = {
                    "fraud_detected": pred == 1,
                    "risk_score": round(prob, 4),
                    "risk_level": get_risk_level(prob),
                    "model_training_stats": MODEL_METRICS[name]
                }
            return {"status": "success", "comparison": results}
        
        selected_model = models.get(model_type, models['xgboost'])
        prediction = int(selected_model.predict(df)[0])
        probability = float(selected_model.predict_proba(df)[0][1])
        
        return {
            "status": "success",
            "model_used": model_type,
            "transaction_approved": prediction == 0,
            "fraud_detected": prediction == 1,
            "risk_score": round(probability, 4),
            "risk_level": get_risk_level(probability),
            "model_training_stats": MODEL_METRICS[model_type]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)