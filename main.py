from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from pathlib import Path

# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time machine learning API to detect fraudulent credit card transactions.",
    version="1.0.0"
)

# Global variables to hold our model and scaler
model = None
scaler = None

# 2. Define the exact JSON schema the API expects using Pydantic
class TransactionRequest(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="The transaction amount in dollars")

@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model_path = BASE_DIR / "fraud_model.pkl"
        scaler_path = BASE_DIR / "scaler.pkl"
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(" Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load assets: {e}")

# 4. The actual REST API Endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Machine learning models are not loaded.")

    try:
        # Convert incoming JSON payload into a dictionary, then to a Pandas DataFrame
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Apply the exact same preprocessing used during training
        df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        
        # Drop the original columns
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        # Enforce column order to match what XGBoost expects (V1-V28, scaled_amount, scaled_time)
        feature_names = model.get_booster().feature_names
        df = df[feature_names]
        
        # Generate prediction
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        return {
            "status": "success",
            "transaction_approved": prediction == 0,
            "fraud_detected": prediction == 1,
            "risk_score": round(probability, 4),
            "message": "Transaction flagged for review." if prediction == 1 else "Transaction approved."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Run the server locally
if __name__ == "__main__":
    # Runs the API on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)