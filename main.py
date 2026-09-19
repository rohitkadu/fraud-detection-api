from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time machine learning API to detect fraudulent credit card transactions.",
    version="1.0.0"
)

model = None
scaler = None

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
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load assets: {e}")

# --- NEW FRONTEND ROUTE ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Fraud Detection Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 text-gray-800 font-sans min-h-screen py-10 px-4">
        <div class="max-w-5xl mx-auto bg-white p-8 rounded-2xl shadow-xl border border-gray-200">
            
            <header class="mb-8 border-b pb-6">
                <h1 class="text-4xl font-extrabold text-indigo-700 mb-2">Machine Learning Fraud Engine</h1>
                <p class="text-lg text-gray-600">A deployable AI pipeline handling highly imbalanced credit card data in real-time.</p>
            </header>

            <section class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
                <div>
                    <h2 class="text-2xl font-bold mb-3">About the Project</h2>
                    <p class="text-gray-700 mb-3">This system uses an <strong>Extreme Gradient Boosting (XGBoost)</strong> model trained on the Kaggle European Credit Card dataset. Out of 284,807 transactions, only 0.17% are fraudulent.</p>
                    <p class="text-gray-700 mb-3">The model specifically optimizes for <strong>Precision-Recall Area Under Curve (PR-AUC)</strong> and heavily penalizes missed frauds using dynamic class weighting.</p>
                    <div class="mt-6 flex gap-4">
                        <a href="/docs" class="bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition shadow-md">Open Swagger UI (Docs)</a>
                    </div>
                </div>
                <div class="bg-indigo-50 p-5 rounded-xl border border-indigo-100">
                    <h3 class="font-bold text-indigo-900 mb-2">Tech Stack</h3>
                    <ul class="list-disc list-inside text-indigo-800 space-y-1">
                        <li><strong>Model:</strong> XGBoost Classifier</li>
                        <li><strong>Data Ops:</strong> Pandas, Scikit-Learn</li>
                        <li><strong>API:</strong> FastAPI, Uvicorn, Pydantic</li>
                        <li><strong>Deployment:</strong> Render.com</li>
                    </ul>
                </div>
            </section>

            <hr class="mb-10">

            <section>
                <h2 class="text-2xl font-bold mb-2">Live API Playground</h2>
                <p class="text-gray-600 mb-6">The JSON below represents a real <strong>stolen credit card transaction</strong>. Click the button to hit the <code class="bg-gray-200 px-2 py-1 rounded">/predict</code> endpoint via a background JavaScript fetch.</p>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Input Block -->
                    <div>
                        <div class="flex justify-between items-center mb-2">
                            <h3 class="font-semibold text-gray-700">Request Body (JSON)</h3>
                        </div>
                        <textarea id="json-input" class="w-full h-80 p-4 font-mono text-sm border rounded-lg bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-inner">{
  "Time": 406.0,
  "V1": -2.312227,
  "V2": 1.951992,
  "V3": -1.609851,
  "V4": 3.997906,
  "V5": -0.522188,
  "V6": -1.426545,
  "V7": -2.537387,
  "V8": 1.391657,
  "V9": -2.770089,
  "V10": -2.772272,
  "V11": 3.202033,
  "V12": -2.899907,
  "V13": -0.595222,
  "V14": -4.289254,
  "V15": 0.389724,
  "V16": -1.140747,
  "V17": -2.830056,
  "V18": -0.016822,
  "V19": 0.416956,
  "V20": 0.126911,
  "V21": 0.517232,
  "V22": -0.035049,
  "V23": -0.465211,
  "V24": 0.320198,
  "V25": 0.044519,
  "V26": 0.177840,
  "V27": 0.261145,
  "V28": -0.143276,
  "Amount": 0.00
}</textarea>
                        <button onclick="testAPI()" class="mt-4 w-full bg-emerald-500 text-white px-6 py-3 rounded-lg font-bold hover:bg-emerald-600 transition shadow-lg">Run Fraud Check</button>
                    </div>
                    
                    <!-- Output Block -->
                    <div>
                        <h3 class="font-semibold text-gray-700 mb-2">JSON Response</h3>
                        <pre id="api-output" class="w-full h-80 p-4 font-mono text-sm border rounded-lg bg-gray-900 text-emerald-400 overflow-auto shadow-inner flex items-center justify-center">Awaiting transaction payload...</pre>
                    </div>
                </div>
            </section>
        </div>

        <script>
            async function testAPI() {
                const output = document.getElementById('api-output');
                const btn = document.querySelector('button[onclick="testAPI()"]');
                
                // Set loading state
                output.innerText = "Connecting to model...";
                output.classList.remove('text-emerald-400', 'text-red-400', 'flex', 'items-center', 'justify-center');
                output.classList.add('text-gray-400');
                btn.innerText = "Processing...";
                btn.disabled = true;
                btn.classList.add('opacity-70', 'cursor-not-allowed');

                try {
                    const payloadText = document.getElementById('json-input').value;
                    const payload = JSON.parse(payloadText);
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    const data = await response.json();
                    
                    // Format and display response
                    output.innerText = JSON.stringify(data, null, 2);
                    
                    // Change color based on fraud detection
                    output.classList.remove('text-gray-400', 'text-emerald-400', 'text-red-400');
                    if(data.fraud_detected) {
                        output.classList.add('text-red-400'); // Red for fraud
                    } else if (data.status === "success") {
                        output.classList.add('text-emerald-400'); // Green for approved
                    } else {
                        output.classList.add('text-yellow-400'); // Yellow for errors (validation)
                    }
                    
                } catch (error) {
                    output.innerText = "Error: Invalid JSON format or API unreachable.";
                    output.classList.remove('text-gray-400', 'text-emerald-400');
                    output.classList.add('text-red-400');
                } finally {
                    // Restore button
                    btn.innerText = "Run Fraud Check";
                    btn.disabled = false;
                    btn.classList.remove('opacity-70', 'cursor-not-allowed');
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

# --- EXISTING PREDICT ROUTE ---
@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Machine learning models are not loaded.")

    try:
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])
        
        df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        feature_names = model.get_booster().feature_names
        df = df[feature_names]
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)