# 🏠 House Price Prediction — ML Project

A complete end-to-end machine learning project for predicting house prices, featuring exploratory data analysis, feature engineering, model comparison (Linear Regression, Random Forest, XGBoost), and an interactive Streamlit web app.

## Project Structure

```
house-price-ml/
├── data/
│   └── generate_data.py      # Synthetic dataset generator
├── src/
│   ├── preprocessing.py      # Data cleaning & preprocessing pipeline
│   ├── feature_engineering.py# Feature creation & selection
│   └── train.py              # Model training & evaluation
├── models/                   # Saved model artifacts (auto-generated)
├── outputs/                  # Plots & metrics (auto-generated)
├── notebooks/
│   └── eda.py                # Exploratory Data Analysis script
├── app.py                    # Streamlit prediction app
└── requirements.txt
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_data.py
```

### 3. Run EDA
```bash
python notebooks/eda.py
```

### 4. Train models
```bash
python src/train.py
```

### 5. Launch Streamlit app
```bash
streamlit run app.py
```

## Models Evaluated
| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted trees (best performer) |

## Metrics
- **RMSE** – Root Mean Squared Error
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination
