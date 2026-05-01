# 📈 S&P 500 Price Direction Prediction

> Binary classification model predicting next-day S&P 500 movement using macroeconomic indicators and technical features — Random Forest achieved **87.5% accuracy and AUC 0.9477**.

---

## 📌 Project Overview

Can macroeconomic data predict short-term stock market direction?  
This project builds a machine learning pipeline to classify whether the S&P 500 will rise more than **0.2%** on the following trading day, using 10 years of financial market and macroeconomic data (2015–2024).

---

## 🎯 Prediction Target

| Label | Definition |
|-------|-----------|
| `1` | Next-day return > +0.2% (Up) |
| `0` | Next-day return ≤ +0.2% (Down / Flat) |

> The 0.2% threshold was selected after testing 0.1%, 0.5%, and other values — 0.2% produced the best model performance.

---

## 🗂 Data Sources

| Source | Variables |
|--------|-----------|
| **yfinance** | S&P 500, VIX (Fear Index), NASDAQ, DXY (Dollar Index), US 10Y / 2Y Treasury Yield, WTI Oil, Copper, Gold, Natural Gas, Corn |
| **FRED API** | CPI, Fed Funds Rate, Unemployment Rate, Retail Sales, PCE, 10Y Inflation Expectation |

**Period**: Jan 2015 – Dec 2024  
**Train / Test Split**: 2015–2022 (train) · 2023–2024 (test) — time-series structure preserved

---

## ⚙️ Feature Engineering

### Economic Indicator Returns
`CPI_change` · `WTI_change` · `Gold_change` · `Inflation_change` · `Retail_change` · `PCE_change` · `Unemp_change`

### Technical Analysis Features
`MA10` · `Above_MA10` · `MACD` · `Momentum_5d` · `Volatility_5d` · `Return_3d_avg`

### Derived Return / Volatility Features
`CumulativeReturn_3d` · `Return_change_1d` · `Volatility_change`

### Market Sentiment Features
`SP_Gold_interaction` (S&P return × Gold change) · `Gold_up_while_SP_down`

### Lag Features
`Retail_change_lag1` · `Unemp_change_lag1`

---

## 🤖 Models & Results

### Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 0.7163 |
| Precision | 0.7179 |
| Recall | 0.6914 |
| **F1 Score** | **0.7044** |
| AUC | 0.79 |

Hyperparameters: `penalty=l1`, `solver=liblinear`, `class_weight=balanced`  
→ Multicollinearity between features limited further improvement. Focus shifted to Random Forest.

---

### ✅ Random Forest (Final Model)

| Metric | Score |
|--------|-------|
| Accuracy | **0.8753** |
| Precision | **0.8787** |
| Recall | **0.8642** |
| **F1 Score** | **0.8714** |
| **AUC** | **0.9477** |
| AP (Avg Precision) | 0.9418 |

Hyperparameters:
```
n_estimators    : 2000
max_depth       : 20
random_state    : 42
min_samples_split: 2
min_samples_leaf : 1
max_features    : None
class_weight    : balanced
```

---

## 🔍 Feature Importance (Random Forest)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `Return_change_1d` | Daily change in return rate |
| 2 | `Return_3d_avg` | 3-day average return |
| 3 | `CumulativeReturn_3d` | 3-day cumulative return |
| 4 | `Volatility_change` | Change in 5-day volatility |
| 5 | `SP_Gold_interaction` | S&P × Gold interaction term |

> Return-based features dominate. Unemployment lag features showed the lowest importance.

---

## 🛠 Tools & Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-black?style=flat)
![FRED](https://img.shields.io/badge/FRED%20API-003087?style=flat)

- **Class imbalance**: Handled with SMOTE
- **Scaling**: StandardScaler
- **Collinearity**: Removed features with correlation > threshold

---

## ⚠️ Limitations

- **No event data**: Fed announcements, geopolitical shocks, and earnings surprises are not reflected
- **Overfitting risk**: Tree-based models with many engineered features are prone to overfitting on historical patterns
- **No confidence output**: Current model returns binary labels only — probability calibration needed for real-world use
- **Static training window**: Model does not retrain as new data arrives

---

## 📁 Repository Structure

```
sp500-price-prediction-ml/
│
├── data/
│   └── sp500_combined.csv        # Merged yfinance + FRED dataset
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_logistic_regression.ipynb
│   └── 04_random_forest.ipynb
│
├── models/
│   └── random_forest_final.pkl
│
├── charts/                        # Confusion matrix, ROC, PR curve
└── README.md
```

---

## 👥 Team

5-person team project — Data Analytics Bootcamp, 2025  
Team name: 머러지지마 (Don't lose to machine learning)
