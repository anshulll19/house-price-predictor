"""
src/database.py — PostgreSQL connection & table initialisation
Uses SQLAlchemy + psycopg2. Connection string read from
st.secrets["database"]["url"].
"""

import streamlit as st
from sqlalchemy import create_engine, text


@st.cache_resource(show_spinner=False)
def get_engine():
    """Return a cached SQLAlchemy engine using st.secrets."""
    url = st.secrets["database"]["url"]
    engine = create_engine(url, pool_pre_ping=True)
    return engine


def init_db():
    """Create the users table if it does not already exist."""
    engine = get_engine()
    create_table_sql = text("""
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
    """)
    with engine.begin() as conn:
        conn.execute(create_table_sql)
