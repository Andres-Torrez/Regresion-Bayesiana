# R# Regression Pipeline: Signal Diagnosis and Model Evaluation

## Project Overview

This project aims to build a complete machine learning pipeline for a regression problem, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing (leakage-safe)
- Baseline modeling
- Model benchmarking
- Residual analysis
- Evaluation and diagnostics

The goal was to predict `quantity_sold` using product, pricing, and contextual features.

---

## Key Outcome

During the development process, it was discovered that the dataset lacks meaningful predictive signal.

As a result, this project is intentionally documented as a **case study of a non-predictive dataset**, highlighting the importance of proper data validation before model optimization.

## Dataset Description

The dataset contains:

- Product information (price, category)
- Customer context (region, payment method)
- Engagement metrics (rating, review count)
- Transaction date (`order_date`)
- Target variable: `quantity_sold`

---

## Critical Observation

The target variable:

- is discrete (values between 1 and 5)
- has low variability
- shows no clear relationship with input features

## Exploratory Data Analysis

EDA revealed several critical issues:

### 1. Target Distribution
- `quantity_sold` takes only a few discrete values (1–5)
- distribution is nearly uniform

### 2. Feature Relationships
- Scatter plots show no visible patterns
- correlations with target are approximately zero

### 3. Categorical Variables
- Mean target value is nearly identical across:
  - product categories
  - customer regions
  - payment methods

### 4. Time Analysis
- No clear temporal trends in demand

---

## Interpretation

These findings strongly suggest that:

> The target variable is not meaningfully related to the available features.

## Data Preprocessing

A robust preprocessing pipeline was implemented:

- Type conversion (dates, numerics)
- Missing value handling
- Feature engineering (time-based features)
- One-hot encoding for categorical variables
- Standardization for numeric features

---

## Leakage Prevention

The following leakage-prone features were removed:

- `discounted_price`
- `total_revenue`

The target variable (`quantity_sold`) was strictly excluded from features.

## Modeling

### Baseline Model

- Model: Bayesian Ridge Regression
- Validation:
  - Time-aware train/test split
  - TimeSeriesSplit cross-validation

### Results

- MAE ≈ 1.20
- RMSE ≈ 1.41
- R² ≈ 0

---

## Interpretation

The model showed:

- No overfitting
- Strong underfitting
- Performance equivalent to predicting a constant value

## Model Benchmarking

The following models were evaluated:

- DummyRegressor (mean baseline)
- Bayesian Ridge
- Random Forest
- HistGradientBoosting

### Results

All models performed similarly:

- R² ≈ 0
- No meaningful improvement over dummy baseline

Random Forest showed signs of overfitting without generalization gains.

## Residual Analysis

Residual diagnostics showed:

- Errors centered around zero (no bias)
- Residual variance ≈ target variance
- No pattern in residual plots
- Predictions collapse around a constant value

---

## Key Insight

> The model is unable to reduce uncertainty compared to a naive baseline.

## Final Conclusion

This dataset does not contain sufficient predictive signal for regression modeling.

Key reasons:

- Target variable is low-cardinality and discrete
- No relationship between features and target
- All models behave like constant predictors

---

## Key Takeaways

This project demonstrates:

- The importance of validating data before modeling
- Why EDA is critical in ML workflows
- How to detect lack of signal
- Why model complexity cannot compensate for poor data

## Next Steps

To build a meaningful regression model, a dataset with:

- continuous target variable
- real feature-target relationships
- sufficient variability

is required.

Future work will involve applying the same pipeline to a more suitable dataset.