<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c3aed&height=200&section=header&text=Customer%20Churn%20Prediction&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=End-to-End%20ML%20Pipeline%20%7C%20Cognifyz%20Technologies&descAlignY=58&descAlign=50" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-7c3aed?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-a855f7?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-06b6d4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-f43f5e?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-10b981?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<br/>

> **Predict customer churn before it happens.** This project builds and benchmarks four machine learning models — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost — on 10,000 real bank customer records to identify churn risk with up to **86.5% accuracy** and **86.0% ROC-AUC**.

<br/>

[![LinkedIn](https://img.shields.io/badge/Karthikeyan_Thirunavukkarasu-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/karthikeyan-thirunavukkarasu-2a2949305)
[![GitHub](https://img.shields.io/badge/karthiikofcl07-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/karthiikofcl07)
[![Internship](https://img.shields.io/badge/Cognifyz_Technologies-Data_Analytics_Intern-7c3aed?style=for-the-badge)](https://cognifyz.com)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Web Application](#-web-application)
- [Results Summary](#-results-summary)
- [Author](#-author)

---

## 🔮 Project Overview

<table>
<tr>
<td width="60%">

Customer churn — when a customer stops using a service — is one of the most costly business problems in subscription-based industries. This project develops a full machine learning pipeline to:

- **Analyze** customer behavior and demographic patterns
- **Identify** the strongest churn predictors
- **Build** and compare four ML classifiers
- **Deploy** a real-time prediction web application
- **Generate** actionable retention strategies

</td>
<td width="40%">

```
📊 Dataset:    10,000 customers
🏦 Industry:  Banking / Finance  
🎯 Target:    Churn (Yes/No)
⚡ Best Model: Random Forest
📈 Accuracy:   86.50%
🏆 ROC-AUC:   86.00%
🔄 CV Folds:   5-Fold Stratified
```

</td>
</tr>
</table>

---

## 📂 Dataset

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Numeric | Customer credit score (350–850) |
| `Geography` | Categorical | Country: France, Germany, Spain |
| `Gender` | Categorical | Male / Female |
| `Age` | Numeric | Customer age (18–92) |
| `Tenure` | Numeric | Years with the bank (0–10) |
| `Balance` | Numeric | Account balance ($) |
| `NumOfProducts` | Numeric | Number of bank products (1–4) |
| `HasCrCard` | Binary | Has credit card (0/1) |
| `IsActiveMember` | Binary | Active in last 6 months (0/1) |
| `EstimatedSalary` | Numeric | Annual salary estimate ($) |
| `Exited` | Binary | **TARGET** — Churned (1) or Retained (0) |

> **Churn Rate:** 20.4% (2,037 churned / 7,963 retained)

---

## ⚙️ Methodology

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Loading   │────▶│   EDA & Viz      │────▶│  Preprocessing   │
│  & Inspection   │     │  10+ charts      │     │  Encode + Scale  │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
┌─────────────────┐     ┌──────────────────┐     ┌────────▼─────────┐
│  Evaluation     │◀────│  Model Training  │◀────│  Train/Test      │
│  ROC, F1, CM    │     │  4 Algorithms    │     │  Split (80/20)   │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

**Preprocessing Steps:**
1. Drop non-predictive columns (`RowNumber`, `CustomerId`, `Surname`)
2. Label Encoding for `Geography` and `Gender`
3. Stratified 80/20 train-test split
4. Standard Scaling applied to Logistic Regression

---

## 🏆 Model Performance

<div align="center">

| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC | CV Score |
|:------|:--------:|:--------:|:---------:|:------:|:-------:|:--------:|
| 🥇 **Random Forest** | **86.50%** | 55.74% | **83.74%** | 41.77% | **86.00%** | 0.8556 ± 0.009 |
| 🥈 Gradient Boosting | 86.00% | **58.08%** | 74.33% | **47.67%** | 85.16% | 0.8538 ± 0.014 |
| 🥉 XGBoost | 85.40% | 56.42% | 71.86% | 46.44% | 84.39% | 0.8527 ± 0.012 |
| Logistic Regression | 80.50% | 22.92% | 58.59% | 14.25% | 77.10% | 0.7495 ± 0.021 |

</div>

> **Winner: Random Forest** — Best accuracy (86.50%) and ROC-AUC (86.00%) with the tightest cross-validation variance (±0.009), making it the most reliable model for production deployment.

---

## 💡 Key Insights

<table>
<tr>
<td width="50%">

### 🔑 Top Churn Predictors
```
1. Age              — Most important feature
2. IsActiveMember   — 2× risk if inactive  
3. Balance          — High balance = more risk
4. NumOfProducts    — 3-4 products = danger zone
5. Geography        — Germany highest risk
6. Gender           — Female customers +8.6%
```

</td>
<td width="50%">

### 📍 Geographic Breakdown
```
Germany  ████████████████ 32.4%
France   ████████         16.2%
Spain    ████████         16.7%

🚨 Germany churns at 2× the rate
   of other regions!
```

</td>
</tr>
<tr>
<td>

### 📊 Churn by Status
```
Inactive members:  ████████████  ~27%
Active members:    █████           ~14%
```
> Re-engagement is the #1 ROI lever

</td>
<td>

### 💰 Balance Paradox
```
High balance (>$150K):   ████████████ ~30%
Medium balance ($80-150K): ████████  ~22%
Zero balance:            ██           ~14%
```
> Wealth ≠ loyalty without personalized service

</td>
</tr>
</table>

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── 📓 notebook/
│   └── Customer_Churn_Prediction.ipynb     ← Full analysis (pre-executed)
│
├── 🌐 app/
│   └── index.html                           ← Live prediction web app
│
├── 📊 Churn_Modelling.csv                   ← Raw dataset (10,000 records)
│
└── 📄 README.md                             ← This file
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/karthiikofcl07/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly jupyter

# Launch Jupyter Notebook
jupyter notebook notebook/Customer_Churn_Prediction.ipynb

# Open Web App
open app/index.html
# or: python -m http.server 8080 (then visit http://localhost:8080/app/)
```

---

## 🌐 Web Application

The project includes a fully functional **ChurnSight** web application featuring:

- **🔮 Live Prediction Engine** — Input any customer profile, get instant churn probability
- **📊 Risk Tier Classification** — Low / Medium / High with visual risk meter
- **🔍 Factor Analysis** — Which specific attributes are driving the risk
- **💡 Retention Recommendations** — Actionable, personalized strategies
- **📈 Model Benchmarks** — Interactive performance comparison dashboard
- **🌙 Dark Theme UI** — Professional, premium design aesthetic

> Open `app/index.html` in any modern browser — no server required!

---

## 📊 Results Summary

```
╔══════════════════════════════════════════════════════════════╗
║           CUSTOMER CHURN PREDICTION — FINAL RESULTS          ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset:        10,000 records | 10 features | 20.4% churn ║
║  Best Model:     Random Forest (n_estimators=200, depth=8)   ║
║  Test Accuracy:  86.50%                                       ║
║  ROC-AUC:        86.00%                                       ║
║  CV Score:       0.8556 ± 0.0092 (5-fold)                    ║
║  Top Predictor:  Age > IsActiveMember > Balance               ║
║  High Risk:      Germany + Female + Inactive + Age 45-55     ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 👤 Author

<div align="center">

<table>
<tr>
<td align="center">
<b>Karthikeyan Thirunavukkarasu</b><br/>
Data Analytics Intern<br/>
Cognifyz Technologies<br/>
<br/>
<a href="https://linkedin.com/in/karthikeyan-thirunavukkarasu-2a2949305">
<img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin"/>
</a>
&nbsp;
<a href="https://github.com/karthiikofcl07">
<img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github"/>
</a>
</td>
</tr>
</table>

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c3aed&height=100&section=footer" width="100%"/>

</div>
