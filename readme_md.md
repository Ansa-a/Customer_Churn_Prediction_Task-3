# 📉 Telco Customer Churn Prediction

**Task 03 — End-to-End Machine Learning Project**  
*Devixo Solutions — AI/ML Internship (Week 3)*

**Author:** Amtul Jamil Ansa  
**Dataset:** [Telco Customer Churn (Kaggle / IBM Sample Data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Stack:** Python, Scikit-Learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib

---

## 📌 Project Overview

Customer churn is a critical issue for telecommunications companies, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project implements an end-to-end machine learning pipeline to identify customers at risk of leaving the service.

By leveraging engineered behavioral features, pipeline-integrated scaling, hyperparameter optimization, and class-balancing strategies, the final model provides reproducible, risk-adjusted churn predictions alongside an interactive web interface.

---

## 🚀 Key Features

* **Data Cleaning & Preprocessing:** Handles implicit null values in `TotalCharges` (new customers with `tenure = 0`), removes duplicate records, and encodes target and service variables cleanly.
* **Feature Engineering:** Creates 6 domain-specific features:
  * `tenure_group`: Binned tenure age buckets (0-12m, 13-24m, 25-48m, 48m+)
  * `avg_monthly_spend`: Average historical monthly spend ($\text{TotalCharges} / \text{tenure}$)
  * `charge_diff`: Difference between current monthly charge and historical average spend
  * `num_services`: Total count of active subscribed services
  * `is_autopay`: Binary indicator for automatic payment methods
  * `has_family`: Binary indicator for partner or dependent coverage
* **Leakage-Free Pipeline:** Integrates `StandardScaler` inside Scikit-Learn `Pipeline` and `ColumnTransformer` instances to guarantee scaling occurs strictly within cross-validation folds.
* **Feature Selection:** Applies correlation thresholds ($\vert{}r\vert{} > 0.90$) and Mutual Information filtering ($MI \ge 0.005$) to select 24 high-signal predictive features.
* **Model Benchmark & Tuning:** Evaluates Decision Trees, Random Forest, SVM, Gradient Boosting, and XGBoost using 5-fold Stratified Cross-Validation (`GridSearchCV` / `RandomizedSearchCV`).
* **Model Deployment & Reusability:** Serializes artifacts via `Joblib` and `Pickle`, supported by a reusable `churn_utils.py` module and an interactive **Streamlit web application**.

---

## 📊 Performance Summary

Models were selected primarily on **ROC-AUC** (threshold-independent ranking score) due to target class imbalance (~26.4% churn rate).

| Model | Baseline CV ROC-AUC | Tuned CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Best)** | 0.8425 | **0.8476** | **0.7922** | **0.6527** | 0.4597 | 0.5394 | **0.8397** |
| **XGBoost** | 0.8177 | 0.8468 | 0.7587 | 0.5295 | 0.7957 | 0.6359 | 0.8400 |
| **Random Forest** | 0.8259 | 0.8460 | 0.7637 | 0.5379 | 0.7634 | 0.6311 | 0.8400 |
| **SVM (RBF Kernel)** | 0.8265 | 0.8400 | 0.7004 | 0.4627 | **0.8172** | 0.5909 | 0.8300 |
| **Decision Tree** | 0.6551 | 0.8299 | 0.7146 | 0.4767 | 0.7984 | 0.5970 | 0.8300 |

> **Key Feature Drivers:** Permutation importance shows `Contract_Month_to_month`, `tenure`, `InternetService_Fiber_optic`, `MonthlyCharges`, and `TotalCharges` as the strongest predictors of churn risk.

---

## 📁 Repository Structure

```text
.
├── churn_utils.py          # Shared data processing & feature engineering logic
├── app.py                  # Streamlit web application interface
├── requirements.txt        # Project dependencies
├── Task_03_Notebook.ipynb  # End-to-end training and evaluation notebook
├── models/
│   ├── best_churn_model.joblib  # Serialized best model pipeline (Joblib)
│   ├── best_churn_model.pkl     # Serialized best model pipeline (Pickle)
│   └── model_metadata.json      # Metadata, feature list, and evaluation metrics
├── model_results_cv.csv    # 5-fold cross-validation metrics report
└── model_results_test.csv  # Final test set evaluation metrics report
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit Prediction App

Launch the local web server:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`. Input custom customer specifications in the sidebar to generate real-time churn probability scores and risk assessments.

---

## 💡 Programmatic Usage

You can load and use the trained model directly in your Python code:

```python
import joblib
import json
import pandas as pd
from churn_utils import prepare

# Load saved pipeline and metadata
model = joblib.load("models/best_churn_model.joblib")
meta = json.load(open("models/model_metadata.json"))

# Sample raw customer input
raw_customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5, "TotalCharges": 191.0
}

# Preprocess & predict
processed_df = prepare(pd.DataFrame([raw_customer]), feature_columns=meta["all_feature_columns"])
churn_prob = model.predict_proba(processed_df[meta["selected_features"]])[0, 1]

print(f"Predicted Churn Probability: {churn_prob:.1%}")
```

---

## 📌 Business Limitations & Future Work

* **Threshold Tuning:** The current default 0.5 classification threshold favors overall precision. Operating thresholds can be tuned based on financial trade-offs (e.g., cost of retention offer vs. Customer Lifetime Value).
* **Model Explainability:** Integrating **SHAP** (SHapley Additive exPlanations) to provide local explanations for individual predictions in the web app.
* **Temporal Dynamics:** Future updates should incorporate time-series transactional features to capture changes in customer behavior over time.