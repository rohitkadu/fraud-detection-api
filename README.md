Absolutely — here is the **complete cleaned-up Markdown file** in one block, ready to copy and paste directly into `README.md`.

````markdown
# 🚀 Rohit's Real-Time Credit Card Fraud Detection Models

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Advanced_ML-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-blue.svg)
![Render](https://img.shields.io/badge/Deployed_on-Render-black.svg)

An end-to-end, production-ready machine learning microservice that detects fraudulent credit card transactions in real time.

This project tackles one of the toughest problems in applied machine learning: **severe class imbalance**. Rather than leaving the models in an exploratory Jupyter notebook, I engineered a full deployment pipeline.

It features a **Shadow Testing architecture**, where multiple models evaluate the same transaction simultaneously, exposed through a fast asynchronous REST API and an interactive frontend.

---

## 🌐 Live Demo & API

- **Interactive UI & Playground:** [View Live Site](https://fraud-detection-api-wld3.onrender.com/)
- **Swagger / OpenAPI Documentation:** [API Docs](https://fraud-detection-api-wld3.onrender.com/docs)
- **Inference Endpoint:** `POST /predict?model_type={xgboost|random_forest|logistic_regression|all}`

---

## 🧠 The Engineering Problem: The Accuracy Paradox

In most machine learning tutorials, classes are evenly split. In this dataset, there are **284,807 transactions**, but only **492 are fraudulent** — a fraud rate of approximately **0.172%**.

A naive model that blindly predicts **"Normal"** for every transaction can achieve approximately **99.83% accuracy**, while failing to detect a single fraudulent transaction.

This demonstrates why accuracy alone is a poor metric for highly imbalanced fraud-detection problems.

To address this, the models are evaluated using metrics that focus on the minority fraud class, particularly **Precision, Recall, F1-Score, and Precision-Recall Area Under Curve (PR-AUC)**.

---

## 🛠️ Data Engineering & Preprocessing

### Dataset

The project uses the **Kaggle European Cardholder Credit Card Fraud Detection dataset** from September 2013.

The dataset contains:

- **284,807 total transactions**
- **492 fraudulent transactions**
- **28 PCA-transformed features:** `V1` - `V28`
- `Time`
- `Amount`
- `Class` — target variable

The PCA-transformed features are anonymized to protect sensitive information.

### Robust Scaling

Standard scaling can be sensitive to extreme financial outliers.

For example, a legitimate transaction with an unusually large amount can significantly affect the mean and standard deviation.

To make preprocessing more robust, I applied **RobustScaler** to the `Amount` and `Time` columns.

RobustScaler uses the **median** and **Interquartile Range (IQR)** instead of the mean and standard deviation, making it less sensitive to extreme values.

### Stratified Partitioning

I used a **stratified 80/20 train/test split** to preserve the class distribution between the training and testing datasets.

This is particularly important for fraud detection because the fraudulent class represents only approximately **0.172%** of the dataset.

Stratification helps ensure that both datasets contain representative proportions of fraudulent and legitimate transactions.

### Zero Data Leakage in Production

The scaler's learned parameters, including the training-set medians and IQRs, are serialized into:

```text
scaler.pkl
````

During production inference, the API uses:

```python
scaler.transform(...)
```

rather than fitting the scaler again.

This ensures that real-time transactions are transformed using the exact preprocessing parameters learned from the training data.

---

## 🤖 Multi-Model Architecture

To simulate an enterprise-style machine learning environment, the system deploys three distinct algorithms.

### 1. XGBoost — Champion Model

**XGBoost** is the primary production model.

It builds decision trees sequentially, with each new tree attempting to correct errors made by previous trees.

Because fraud is extremely rare compared with legitimate transactions, I configured a dynamic `scale_pos_weight` of approximately **577**.

This gives significantly more importance to fraudulent examples during training.

The approximate ratio is:

```text
284,315 legitimate transactions
-------------------------------- ≈ 577
492 fraudulent transactions
```

This helps the model focus more heavily on identifying the minority fraud class.

---

### 2. Random Forest — Ensemble Baseline

**Random Forest** is used as an ensemble baseline.

It creates multiple decision trees independently and combines their predictions.

This provides a useful comparison against the sequential boosting approach used by XGBoost.

---

### 3. Logistic Regression — Statistical Baseline

**Logistic Regression** provides a simple and computationally efficient baseline.

It demonstrates how a relatively simple linear model behaves when dealing with an extremely imbalanced classification problem.

In this project, Logistic Regression achieves high fraud recall but produces a much larger number of false positives, resulting in substantially lower fraud precision.

---

## 📊 Performance Evaluation

The following results were obtained on the project's test set:

| Model                   |   Accuracy | Precision (Fraud) | Recall (Fraud) |  F1-Score |
| ----------------------- | ---------: | ----------------: | -------------: | --------: |
| **XGBoost**             | **99.96%** |         **94.5%** |      **85.2%** | **89.6%** |
| **Random Forest**       |     99.95% |             92.1% |          78.4% |     84.7% |
| **Logistic Regression** |     97.46% |              6.5% |          91.0% |     12.2% |

### Metric Definitions

**Accuracy**

The percentage of all transactions classified correctly.

```text
Correct Predictions
------------------- × 100
Total Predictions
```

Accuracy can be misleading for fraud detection because legitimate transactions vastly outnumber fraudulent transactions.

**Precision**

Of the transactions predicted as fraudulent, precision measures how many were actually fraudulent.

```text
True Positives
------------------------------
True Positives + False Positives
```

Higher precision means fewer legitimate customers are incorrectly flagged as fraudulent.

**Recall**

Recall measures how many actual fraudulent transactions were successfully detected.

```text
True Positives
-------------------------
True Positives + False Negatives
```

Higher recall means fewer fraudulent transactions are missed.

**F1-Score**

F1-score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

**PR-AUC**

Precision-Recall Area Under Curve is particularly useful for evaluating classification systems with severe class imbalance because it focuses on the relationship between precision and recall for the minority class.

---

## 🏗️ System Architecture

The application follows an end-to-end inference pipeline:

```text
                   ┌──────────────────────┐
                   │     User / Client    │
                   └──────────┬───────────┘
                              │
                              │ JSON Request
                              ▼
                   ┌──────────────────────┐
                   │      FastAPI API     │
                   │   Pydantic Validation│
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │    Data Preprocessing│
                   │     RobustScaler     │
                   └──────────┬───────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      ┌────────────┐   ┌────────────┐   ┌──────────────┐
      │  XGBoost   │   │Random Forest│   │   Logistic   │
      │   Model    │   │    Model    │   │  Regression  │
      └──────┬─────┘   └──────┬─────┘   └───────┬──────┘
             │                │                  │
             └────────────────┼──────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Fraud Probability /  │
                   │     Risk Score       │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Risk Classification │
                   │ LOW / MEDIUM / HIGH  │
                   │       / CRITICAL     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │     JSON Response    │
                   └──────────────────────┘
```

---

## 💻 Tech Stack

### Machine Learning

* **XGBoost**
* **Scikit-Learn**
* **Pandas**
* **NumPy**

### Backend API

* **FastAPI**
* **Uvicorn**
* **Pydantic**

Pydantic is used for strict request-schema validation.

### Frontend

* **HTML5**
* **Tailwind CSS**
* **JavaScript**
* **Fetch API**

### Model Serialization

* **Joblib**

### Deployment

* **Render**

---

## 📁 Project Structure

A typical project structure looks like this:

```text
fraud-detection-api/
│
├── main.py
├── requirements.txt
├── scaler.pkl
│
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── logistic_regression_model.pkl
│
├── static/
│   ├── index.html
│   ├── script.js
│   └── ...
│
└── README.md
```

> File names may vary depending on the final repository implementation.

---

## 🚀 Local Installation & Usage

Want to run this project locally?

Follow the steps below.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-api.git
cd fraud-detection-api
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv fraud_env
fraud_env\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv fraud_env
source fraud_env/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Start the FastAPI Server

```bash
python main.py
```

The application should start on:

```text
http://localhost:8000
```

Open your browser and navigate to:

```text
http://localhost:8000
```

to access the interactive dashboard.

---

## 📡 API Reference

### Evaluate a Transaction

```http
POST /predict
```

The endpoint accepts transaction information and returns a fraud prediction, risk score, risk level, and model information.

---

### Query Parameters

| Parameter    | Type   | Description               | Default   |
| ------------ | ------ | ------------------------- | --------- |
| `model_type` | string | Model used for prediction | `xgboost` |

Supported values:

```text
xgboost
random_forest
logistic_regression
all
```

Example:

```http
POST /predict?model_type=xgboost
```

To evaluate the transaction using all available models:

```http
POST /predict?model_type=all
```

---

## 📥 Sample Request Payload

```json
{
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
}
```

---

## 📤 Sample Response

```json
{
  "status": "success",
  "model_used": "xgboost",
  "transaction_approved": false,
  "fraud_detected": true,
  "risk_score": 0.9841,
  "risk_level": "CRITICAL",
  "model_training_stats": {
    "accuracy": "99.96%",
    "precision": "94.5%",
    "recall": "85.2%",
    "f1_score": "89.6%"
  }
}
```

---

## 🔍 Understanding the Response

### `status`

Indicates whether the API request was processed successfully.

Example:

```json
"status": "success"
```

### `model_used`

Indicates which machine learning model generated the prediction.

Example:

```json
"model_used": "xgboost"
```

### `transaction_approved`

Boolean value indicating whether the transaction passed the configured fraud-detection decision.

```json
"transaction_approved": false
```

### `fraud_detected`

Indicates whether the model classified the transaction as fraudulent.

```json
"fraud_detected": true
```

### `risk_score`

Represents the model's estimated fraud probability or risk score.

Example:

```json
"risk_score": 0.9841
```

### `risk_level`

A human-readable representation of the calculated transaction risk.

Possible levels include:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

---

## 🧪 Shadow Testing

One of the key architectural ideas in this project is **Shadow Testing**.

Instead of relying exclusively on one model during experimentation, the API can evaluate the same transaction with multiple models.

For example:

```http
POST /predict?model_type=all
```

can run:

```text
                    Transaction
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      XGBoost      Random Forest   Logistic Regression
          │              │              │
          ▼              ▼              ▼
      Prediction     Prediction     Prediction
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Model Comparison
```

This architecture makes it possible to compare model behavior on the same transaction and observe differences in fraud probability and classification.

---

## ⚖️ Why Accuracy Alone Is Not Enough

The extreme class imbalance makes accuracy an unreliable standalone metric.

Suppose a model receives:

```text
284,807 transactions
```

and predicts:

```text
Normal
```

for every transaction.

Because only:

```text
492
```

transactions are fraudulent, the model would still achieve approximately:

```text
99.83% accuracy
```

despite detecting:

```text
0 fraudulent transactions
```

This is why fraud detection systems should consider multiple metrics, including:

* Precision
* Recall
* F1-Score
* PR-AUC
* Confusion Matrix
* False Positive Rate
* False Negative Rate

---

## 📈 Key Machine Learning Insights

### Severe Class Imbalance

Fraud represents only a tiny fraction of the overall dataset.

```text
Fraudulent Transactions: 492
Total Transactions:      284,807

Fraud Rate ≈ 0.172%
```

This makes conventional accuracy-based optimization insufficient.

### Precision vs Recall

Fraud detection involves a trade-off between:

```text
High Recall
     │
     │ Detect more fraud
     │
     ▼
More False Positives
```

and:

```text
High Precision
     │
     │ Reduce false alarms
     │
     ▼
Potentially More Missed Fraud
```

The appropriate operating threshold depends on the business requirements of the fraud-detection system.

---

## 🔐 Production Considerations

This project demonstrates a production-oriented ML API architecture, but a real financial fraud system would require additional controls before being used for actual payment decisions.

Potential production improvements include:

* Authentication and authorization
* HTTPS/TLS
* Rate limiting
* Request logging
* Monitoring and alerting
* Model versioning
* Data drift detection
* Feature drift detection
* Model performance monitoring
* Automated retraining pipelines
* Threshold optimization
* Explainable AI
* Audit logging
* Secure secret management
* Database integration
* High-availability infrastructure
* Containerization
* CI/CD pipelines

---

## 🧩 Future Improvements

Potential future enhancements include:

* [ ] Add PR-AUC metrics to the API response
* [ ] Add ROC-AUC comparison
* [ ] Add confusion-matrix visualization
* [ ] Add model explainability using SHAP
* [ ] Add transaction history features
* [ ] Add automated model retraining
* [ ] Add data-drift monitoring
* [ ] Add model versioning
* [ ] Add Docker support
* [ ] Add CI/CD using GitHub Actions
* [ ] Add authentication and API keys
* [ ] Add Redis-based caching
* [ ] Add PostgreSQL transaction logging
* [ ] Add comprehensive unit and integration tests
* [ ] Add production monitoring with Prometheus/Grafana

---

## 🌐 Live Deployment

### Interactive Dashboard

**Live Application:**

[https://fraud-detection-api-wld3.onrender.com/](https://fraud-detection-api-wld3.onrender.com/)

### API Documentation

**Swagger UI:**

[https://fraud-detection-api-wld3.onrender.com/docs](https://fraud-detection-api-wld3.onrender.com/docs)

The Swagger interface can be used to inspect available endpoints and interactively test the API.

---

## 📚 Dataset

This project uses the **Credit Card Fraud Detection** dataset containing European cardholder transactions from September 2013.

The dataset is commonly used for benchmarking fraud-detection algorithms and contains anonymized PCA-transformed features.

### Important Dataset Characteristics

```text
Total Transactions:       284,807
Fraudulent Transactions:      492
Legitimate Transactions: 284,315
Fraud Rate:                 ~0.172%
```

The `V1` through `V28` features are PCA-transformed and anonymized.

---

## ⚠️ Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

The model predictions should not be treated as guaranteed fraud determinations or as a replacement for a production financial fraud-detection system.

Real-world payment systems require extensive validation, monitoring, security controls, regulatory compliance, human review processes, and continuously updated models.

---

## 👨‍💻 Author

**Rohit**

Machine Learning & Backend Engineering Project

Built with:

* Python
* XGBoost
* Scikit-Learn
* FastAPI
* Pandas
* NumPy
* Pydantic
* Tailwind CSS
* JavaScript
* Render

---

## ⭐ Project Highlights

```text
✅ End-to-end ML pipeline
✅ Severe class-imbalance handling
✅ RobustScaler preprocessing
✅ Stratified train/test split
✅ XGBoost fraud detection
✅ Random Forest baseline
✅ Logistic Regression baseline
✅ FastAPI REST API
✅ Pydantic request validation
✅ Real-time inference
✅ Multi-model shadow testing
✅ Interactive frontend
✅ Serialized production models
✅ Render deployment
✅ Swagger/OpenAPI documentation
```

---

## 🚀 Final Takeaway

This project demonstrates how a machine learning model can be taken beyond a notebook and integrated into a deployable real-time application.

The core challenge is not simply achieving high accuracy. It is building a system that can identify a very small number of fraudulent transactions while controlling false positives and maintaining a reliable production inference pipeline.

The combination of:

```text
Data Engineering
       +
Imbalanced Classification
       +
Model Comparison
       +
Robust Preprocessing
       +
FastAPI
       +
Real-Time Inference
       +
Cloud Deployment
```

creates a complete machine learning application rather than an isolated model-training experiment.

---

⭐ **If you found this project useful, consider giving the repository a star!**

```
```
