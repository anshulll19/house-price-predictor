"""
src/database.py — PostgreSQL connection & table initialisation
Uses SQLAlchemy + psycopg2. Connection string read from
st.secrets["database"]["url"].
"""

import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import create_engine, text


@st.cache_resource(show_spinner=False)
def get_engine():
    """Return a cached SQLAlchemy engine using st.secrets."""
    url = st.secrets["database"]["url"]
    engine = create_engine(url, pool_pre_ping=True)
    return engine


def init_db():
    """Create all required tables if they do not already exist."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id               SERIAL PRIMARY KEY,
                username         VARCHAR(80)  NOT NULL UNIQUE,
                email            VARCHAR(255) NOT NULL UNIQUE,
                hashed_password  TEXT         NOT NULL,
                created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                last_login       TIMESTAMPTZ,
                failed_attempts  INTEGER      NOT NULL DEFAULT 0,
                is_locked        BOOLEAN      NOT NULL DEFAULT FALSE
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              SERIAL PRIMARY KEY,
                user_id         INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                city            VARCHAR(100) NOT NULL,
                locality        VARCHAR(100) NOT NULL,
                area_sqft       INTEGER      NOT NULL,
                bhk             INTEGER      NOT NULL,
                predicted_price NUMERIC(18,2) NOT NULL,
                created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
        """))


# ── Prediction helpers ────────────────────────────────────────────────────────

def save_prediction(user_id: int, city: str, locality: str,
                    area_sqft: int, bhk: int, predicted_price: float) -> bool:
    """Insert a prediction record. Returns True on success."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO predictions (user_id, city, locality, area_sqft, bhk, predicted_price)
                VALUES (:user_id, :city, :locality, :area_sqft, :bhk, :predicted_price)
            """), {
                "user_id": user_id,
                "city": city,
                "locality": locality,
                "area_sqft": int(area_sqft),
                "bhk": int(bhk),
                "predicted_price": float(predicted_price),
            })
        return True
    except Exception:
        return False


def get_user_predictions(user_id: int) -> list[dict]:
    """Return all predictions for a user, newest first."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, city, locality, area_sqft, bhk, predicted_price, created_at
                FROM   predictions
                WHERE  user_id = :user_id
                ORDER BY created_at DESC
            """), {"user_id": user_id}).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def delete_prediction(pred_id: int, user_id: int) -> bool:
    """Delete a prediction by id, only if it belongs to user_id."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM predictions
                WHERE id = :pred_id AND user_id = :user_id
            """), {"pred_id": pred_id, "user_id": user_id})
        return True
    except Exception:
        return False


def get_user_stats(user_id: int) -> dict:
    """Return total predictions count and average predicted price."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) AS total,
                       COALESCE(AVG(predicted_price), 0) AS avg_price
                FROM   predictions
                WHERE  user_id = :user_id
            """), {"user_id": user_id}).fetchone()
        return {"total": int(row.total), "avg_price": float(row.avg_price)}
    except Exception:
        return {"total": 0, "avg_price": 0.0}


def get_user_profile(user_id: int) -> dict:
    """Return profile info: username, email, created_at, total predictions."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT u.username, u.email, u.created_at,
                       COUNT(p.id) AS total_predictions
                FROM   users u
                LEFT JOIN predictions p ON p.user_id = u.id
                WHERE  u.id = :user_id
                GROUP BY u.username, u.email, u.created_at
            """), {"user_id": user_id}).fetchone()
        if row:
            return dict(row._mapping)
        return {}
    except Exception:
        return {}
