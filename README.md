# 🏠 India House Price Estimator

An end-to-end machine learning web application for predicting residential property prices across 10 major Indian cities. Built with XGBoost, Streamlit, PostgreSQL, and a futuristic glassmorphism UI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## ✨ Features

### 🔐 Authentication
- Secure registration & login with **bcrypt** password hashing
- Account lockout after 5 failed attempts
- Session persistence via `st.session_state`
- Smooth logout that clears all session data

### 📊 Personal Dashboard
- **Metric tiles** — total predictions, average price, cities explored
- **Line chart** — prediction price history over time
- **Bar chart** — prediction count by city
- **History table** — all past predictions with one-click delete

### 👤 User Profile
- Displays username, email, member since date, total predictions

### 🏠 Price Estimator
- Covers **10 cities**: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, Kolkata, Ahmedabad, Noida, Jaipur
- Inputs: carpet area, BHK, bathrooms, floor, furnishing, parking, lift, east-facing, property age
- Confidence interval ±12% with market insight
- Every estimate is **automatically saved** to the database

### 📈 Analytics Tabs
| Tab | Content |
|-----|---------|
| 🏙️ City Prices | Median price/sqft bar chart per city |
| 📈 Value Drivers | Illustrative factor attribution chart |
| 🤖 Model Performance | RMSE, MAE, R² for all trained models |
| 🗺️ Market Data | Interactive scatter & box plots from the dataset |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost (+ Linear Regression, Random Forest comparison) |
| Frontend | Streamlit 1.33 · Plotly · custom glassmorphism CSS |
| Database | PostgreSQL (Neon DB) via SQLAlchemy + psycopg2 |
| Auth | bcrypt · `st.session_state` |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
house-price-ml/
├── data/
│   └── generate_data.py        # Synthetic dataset generator
├── src/
│   ├── auth.py                 # Registration, login, session management
│   ├── database.py             # DB engine, users & predictions tables, helpers
│   ├── preprocessing.py        # Data cleaning & preprocessing pipeline
│   ├── feature_engineering.py  # Feature creation & selection
│   └── train.py                # Model training & evaluation
├── models/                     # Saved model artifacts (auto-generated)
├── outputs/                    # Metrics & plots (auto-generated)
├── notebooks/
│   └── eda.py                  # Exploratory Data Analysis
├── .streamlit/
│   └── secrets.toml            # DB connection string (not committed)
├── app.py                      # Main Streamlit application
├── requirements.txt
└── runtime.txt
```

---

## 🚀 Quickstart

### 1. Clone & install
```bash
git clone https://github.com/your-username/house-price-ml.git
cd house-price-ml
pip install -r requirements.txt
```

### 2. Configure secrets
Create `.streamlit/secrets.toml`:
```toml
[database]
url = "postgresql://user:password@host/dbname"
```

### 3. Generate dataset & train
```bash
python data/generate_data.py
python notebooks/eda.py       # optional EDA
python src/train.py
```

### 4. Run locally
```bash
streamlit run app.py
```

---

## 🗄️ Database Schema

```sql
-- Users table
CREATE TABLE users (
    id               SERIAL PRIMARY KEY,
    username         VARCHAR(80)  NOT NULL UNIQUE,
    email            VARCHAR(255) NOT NULL UNIQUE,
    hashed_password  TEXT         NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login       TIMESTAMPTZ,
    failed_attempts  INTEGER      NOT NULL DEFAULT 0,
    is_locked        BOOLEAN      NOT NULL DEFAULT FALSE
);

-- Predictions table
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    city            VARCHAR(100)  NOT NULL,
    locality        VARCHAR(100)  NOT NULL,
    area_sqft       INTEGER       NOT NULL,
    bhk             INTEGER       NOT NULL,
    predicted_price NUMERIC(18,2) NOT NULL,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
```

> Tables are created automatically on first run via `init_db()`.

---

## 🤖 Models Evaluated

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| **XGBoost** | **Gradient-boosted trees — best performer** |

**Metrics**: RMSE · MAE · R² (evaluated on 20% held-out test set)

---

## ☁️ Deploying to Streamlit Cloud

1. Push repo to GitHub
2. Connect to [Live Demo: https://house-price-predictor-g9knurrqope3nuekhrraep.streamlit.app]
3. Set `Main file path` → `app.py`
4. Add `database.url` under **Secrets** in the dashboard
5. Deploy 🚀
