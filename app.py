"""
app.py — India House Price Estimator
Futuristic Glassmorphism UI · Dark Theme · Streamlit 1.55+
Run: streamlit run app.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import json, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.feature_engineering import engineer_features, CITY_TIER_MAP
from src.database import (
    init_db, save_prediction, get_user_predictions,
    delete_prediction, get_user_stats, get_user_profile,
)
from src.auth import (
    register_user, login_user, logout_user,
    is_authenticated, get_current_user, _set_session,
)

st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CITIES = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai",
          "Pune","Kolkata","Ahmedabad","Noida","Jaipur"]
CITY_BASE_PPSF = {
    "Mumbai":18000,"Delhi":12000,"Bangalore":9500,"Hyderabad":7500,
    "Chennai":7000,"Pune":7200,"Kolkata":5500,"Ahmedabad":5000,
    "Noida":6000,"Jaipur":4500,
}
CITY_COORDS = {
    "Mumbai":(19.08,72.88),"Delhi":(28.70,77.10),"Bangalore":(12.97,77.59),
    "Hyderabad":(17.38,78.49),"Chennai":(13.08,80.27),"Pune":(18.52,73.86),
    "Kolkata":(22.57,88.36),"Ahmedabad":(23.02,72.57),"Noida":(28.54,77.39),
    "Jaipur":(26.91,75.79),
}
LOCALITY_OPTS   = ["Premium","Mid","Budget"]
FURNISHING_OPTS = ["Unfurnished","Semi-Furnished","Fully Furnished"]
ROOT = Path(__file__).parent


def fmt_inr(value: float, compact: bool = False) -> str:
    if value >= 1_00_00_000:
        return f"₹{value/1_00_00_000:.2f}Cr" if compact else f"₹{value/1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"₹{value/1_00_000:.1f}L" if compact else f"₹{value/1_00_000:.2f} L"
    return f"₹{value:,.0f}"

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=11),
    margin=dict(l=8,r=8,t=36,b=8),
    title_font=dict(size=12, color="#e2e8f0", family="Space Grotesk"),
)

@st.cache_resource(show_spinner=False)
def load_model():
    p = ROOT/"models"/"best_model.joblib"
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_metrics():
    p = ROOT/"outputs"/"metrics.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_city_stats():
    p = ROOT/"outputs"/"city_stats.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_market_data():
    p = ROOT/"data"/"housing.csv"
    return pd.read_csv(p) if p.exists() else None

def run_prediction(model, inputs: dict) -> float | None:
    import src.preprocessing as pm
    from src.feature_engineering import ENGINEERED_NUMERIC_FEATURES
    pm.NUMERIC_FEATURES = ENGINEERED_NUMERIC_FEATURES
    row = engineer_features(pd.DataFrame([inputs]))
    return float(model.predict(row)[0])


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: #020817;
    min-height: 100vh;
}
.main .block-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px 80px 32px;
}
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#3b82f6,#8b5cf6); border-radius: 99px; }

/* ── Particle canvas ── */
#particle-canvas {
    position: fixed; top:0; left:0; width:100%; height:100%;
    pointer-events: none; z-index: 0;
}

/* ── Glass card base ── */
.glass {
    background: rgba(15,23,42,0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 60px rgba(99,102,241,0.2), 0 0 0 1px rgba(139,92,246,0.3);
}
.glass::before {
    content:'';
    position:absolute; inset:0;
    border-radius:16px;
    padding:1px;
    background: linear-gradient(135deg, rgba(59,130,246,0.5), rgba(139,92,246,0.3), rgba(236,72,153,0.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events:none;
    animation: borderPulse 4s ease-in-out infinite;
}
@keyframes borderPulse {
    0%,100% { opacity:0.6; }
    50% { opacity:1; }
}

/* ── Neon glow pulse ── */
.glow-card {
    box-shadow: 0 0 20px rgba(59,130,246,0.15), 0 8px 32px rgba(0,0,0,0.4);
    animation: cardGlow 3s ease-in-out infinite;
}
@keyframes cardGlow {
    0%,100% { box-shadow: 0 0 20px rgba(59,130,246,0.15), 0 8px 32px rgba(0,0,0,0.4); }
    50% { box-shadow: 0 0 40px rgba(139,92,246,0.3), 0 8px 48px rgba(0,0,0,0.5); }
}

/* ── Result card float ── */
.float-card {
    animation: floatY 4s ease-in-out infinite;
}
@keyframes floatY {
    0%,100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

/* ── Hero section ── */
.hero {
    padding: 32px 0 24px 0;
    position: relative;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    letter-spacing: -0.03em;
    animation: fadeSlideDown 0.8s ease both;
}
@keyframes fadeSlideDown {
    from { opacity:0; transform:translateY(-20px); }
    to { opacity:1; transform:translateY(0); }
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-top: 10px;
    line-height: 1.6;
    animation: fadeSlideDown 0.8s 0.2s ease both;
}

/* ── XGBoost badge ── */
.xgb-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 6px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: #a78bfa;
    animation: badgePop 0.6s 0.5s cubic-bezier(0.34,1.56,0.64,1) both, badgeGlow 3s 1s ease-in-out infinite;
    cursor: default;
}
@keyframes badgePop {
    from { opacity:0; transform:scale(0.5); }
    to { opacity:1; transform:scale(1); }
}
@keyframes badgeGlow {
    0%,100% { box-shadow: 0 0 8px rgba(99,102,241,0.2); }
    50% { box-shadow: 0 0 20px rgba(139,92,246,0.5); }
}
.badge-dot {
    width:7px; height:7px;
    background: #a78bfa;
    border-radius: 50%;
    box-shadow: 0 0 8px #a78bfa;
    animation: dotPulse 1.5s ease-in-out infinite;
}
@keyframes dotPulse {
    0%,100% { transform:scale(1); opacity:1; }
    50% { transform:scale(1.6); opacity:0.6; }
}

/* ── Section headers ── */
.sec-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    margin-top: 20px;
    margin-bottom: 14px;
}

/* ── Staggered slide-in inputs ── */
div[data-testid="stSelectbox"],
div[data-testid="stNumberInput"],
div[data-testid="stToggle"] {
    animation: slideInLeft 0.5s ease both;
}
@keyframes slideInLeft {
    from { opacity:0; transform:translateX(-24px); }
    to { opacity:1; transform:translateX(0); }
}

/* ── Input styling ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    height: 42px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2), 0 0 16px rgba(99,102,241,0.15) !important;
    outline:none !important;
}
div[data-testid="stNumberInput"] button {
    background: rgba(30,41,59,0.8) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #94a3b8 !important;
    transition: all 0.15s !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: rgba(99,102,241,0.2) !important;
    color: #a78bfa !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.3) !important;
}
div[data-testid="stNumberInput"] button:active {
    animation: ripple 0.4s ease;
}
@keyframes ripple {
    0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.6); }
    100% { box-shadow: 0 0 0 12px rgba(99,102,241,0); }
}
label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}
div[data-testid="stToggle"] label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Toggle switches ── */
div[data-testid="stToggle"] input[type="checkbox"] { accent-color: #6366f1; }
div[data-testid="stToggle"] > label > div:first-child {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    transition: all 0.3s ease !important;
}

/* ── CTA Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899) !important;
    background-size: 200% 200% !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.8rem 1.5rem !important;
    margin-top: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.3) !important;
    animation: gradientShift 4s ease infinite !important;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 40px rgba(99,102,241,0.5), 0 8px 32px rgba(0,0,0,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.res-card {
    background: rgba(15,23,42,0.7);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 32px 28px;
    min-height: 320px;
    position: relative;
    overflow: hidden;
    animation: floatY 4s ease-in-out infinite, cardGlow 3s ease-in-out infinite;
}
.res-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    animation: gradientSlide 3s linear infinite;
}
@keyframes gradientSlide {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}
.res-card::after {
    content:'';
    position:absolute; bottom:-50px; right:-50px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.res-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 12px;
}
.res-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    line-height: 1;
    text-shadow: none;
    filter: drop-shadow(0 0 20px rgba(99,102,241,0.4));
}
.res-context {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.range-row {
    display: flex; gap: 8px; margin-top: 24px;
}
.range-cell {
    flex:1;
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
    transition: all 0.2s;
}
.range-cell:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(99,102,241,0.2);
}
.range-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 4px;
}
.range-val { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 600; color: #94a3b8; }
.range-val.accent { color: #a78bfa; text-shadow: 0 0 10px rgba(167,139,250,0.5); }
.conf-tag {
    display: inline-block;
    margin-top: 16px;
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    padding: 4px 12px;
}
.insight-wrap {
    margin-top: 16px;
    padding: 12px 16px;
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Empty state ── */
.empty-card {
    background: rgba(15,23,42,0.4);
    backdrop-filter: blur(16px);
    border: 1px dashed rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 60px 24px;
    min-height: 300px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 14px; opacity: 0.5; animation: floatY 3s ease-in-out infinite; }
.empty-title { font-family:'Space Grotesk',sans-serif; font-size:0.9rem; font-weight:500; color:#475569; }
.empty-hint  { font-size:0.75rem; color:#334155; margin-top:6px; }

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(99,102,241,0.15);
    gap:0;
    position: relative;
}
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: color 0.2s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #6366f1 !important;
    text-shadow: 0 0 10px rgba(99,102,241,0.5) !important;
}
button[data-baseweb="tab"]:hover { color: #94a3b8 !important; }

/* ── Chart helpers ── */
.chart-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 2px;
}
.chart-desc { font-size: 0.75rem; color: #475569; margin-bottom: 14px; }

/* ── Perf tiles ── */
.perf-tile {
    background: rgba(15,23,42,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 18px 20px;
    transition: all 0.3s ease;
}
.perf-tile:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.perf-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin-bottom: 6px;
}
.perf-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #a78bfa;
    text-shadow: 0 0 12px rgba(167,139,250,0.4);
}
.perf-sub { font-size: 0.68rem; color: #334155; margin-top: 3px; }

/* ── Success/Error ── */
div[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
    color: #34d399 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stError"] {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar (live summary) ── */
section[data-testid="stSidebar"] {
    background: rgba(10,15,28,0.85) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6366f1; margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}
.sb-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(30,41,59,0.5);
}
.sb-label { font-size: 0.72rem; color: #475569; }
.sb-value { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #a78bfa; font-weight: 500; }

/* ── Loading spinner override ── */
div[data-testid="stSpinner"] > div {
    border-color: rgba(99,102,241,0.3) !important;
    border-top-color: #6366f1 !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 10px; overflow: hidden;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

/* ── Auth page ── */
.auth-wrap {
    max-width: 460px;
    margin: 80px auto 0 auto;
}
.auth-card {
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 40px 36px 36px 36px;
    position: relative;
    overflow: hidden;
    animation: fadeSlideDown 0.6s ease both;
}
.auth-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}
.auth-logo {
    font-size: 2.8rem;
    text-align: center;
    margin-bottom: 4px;
}
.auth-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.auth-sub {
    font-size: 0.78rem;
    color: #475569;
    text-align: center;
    margin-bottom: 28px;
    font-family: 'Inter', sans-serif;
}
/* Style text inputs inside auth card */
div[data-testid="stTextInput"] input {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}
/* Sidebar user card */
.sb-user-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 18px;
}
.sb-user-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 3px;
}
.sb-user-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    line-height: 1.7;
}

/* ── Dashboard metric tiles ── */
.dash-metric {
    background: rgba(15,23,42,0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 14px;
    padding: 20px 22px;
    transition: all 0.3s ease;
}
.dash-metric:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
.dash-metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #475569; margin-bottom: 8px;
}
.dash-metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #a78bfa;
    text-shadow: 0 0 14px rgba(167,139,250,0.35);
    line-height: 1;
}
.dash-metric-sub {
    font-size: 0.65rem; color: #334155; margin-top: 5px;
}

/* ── Profile card ── */
.profile-card {
    background: rgba(15,23,42,0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    position: relative; overflow: hidden;
}
.profile-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}
.profile-avatar {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 16px;
    box-shadow: 0 0 24px rgba(99,102,241,0.4);
}
.profile-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    margin-bottom: 4px;
}
.profile-email {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #475569; margin-bottom: 20px;
}
.profile-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(30,41,59,0.5);
}
.profile-stat-lbl { font-size: 0.75rem; color: #475569; }
.profile-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #a78bfa; font-weight: 500;
}

/* ── Delete button ── */
.stButton > button[kind="secondary"] {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    color: #f87171 !important;
    border-radius: 8px !important;
    font-size: 0.75rem !important;
    padding: 0.3rem 0.8rem !important;
    margin-top: 0 !important;
    width: auto !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(239,68,68,0.2) !important;
    border-color: rgba(239,68,68,0.6) !important;
    transform: translateY(-1px) !important;
}
</style>
"""

PARTICLE_JS = """
<canvas id="particle-canvas"></canvas>
<script>
(function(){
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');
let W, H, particles = [];
const N = 80, COLOR = 'rgba(99,102,241,', MAX_DIST = 130;
function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; }
window.addEventListener('resize', resize); resize();
for(let i=0;i<N;i++) particles.push({
    x:Math.random()*W, y:Math.random()*H,
    vx:(Math.random()-0.5)*0.4, vy:(Math.random()-0.5)*0.4,
    r:Math.random()*2+1
});
function draw(){
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<N;i++){
        const p=particles[i];
        p.x+=p.vx; p.y+=p.vy;
        if(p.x<0||p.x>W) p.vx*=-1;
        if(p.y<0||p.y>H) p.vy*=-1;
        ctx.beginPath();
        ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle=COLOR+'0.6)';
        ctx.fill();
        for(let j=i+1;j<N;j++){
            const q=particles[j];
            const d=Math.hypot(p.x-q.x,p.y-q.y);
            if(d<MAX_DIST){
                ctx.beginPath();
                ctx.strokeStyle=COLOR+(1-d/MAX_DIST)*0.3+')';
                ctx.lineWidth=0.5;
                ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(draw);
}
draw();
})();
</script>
"""


def inject_css():
    st.markdown(CSS + PARTICLE_JS, unsafe_allow_html=True)


def render_sidebar(city, loc, area, bhk, baths, age, t_fl, floor, furn, park, lift, east):
    with st.sidebar:
        # ── Logged-in user card ────────────────────────────────────────────
        user = get_current_user()
        if user:
            last_login = user.get("last_login")
            created_at = user.get("created_at")
            if last_login and hasattr(last_login, "strftime"):
                ll_str = last_login.strftime("%d %b %Y, %H:%M")
            else:
                ll_str = "This session"
            if created_at and hasattr(created_at, "strftime"):
                joined_str = created_at.strftime("%d %b %Y")
            else:
                joined_str = "—"
            st.markdown(
                f'<div class="sb-user-card">'
                f'<div class="sb-user-name">👤 {user["username"]}</div>'
                f'<div class="sb-user-meta">'
                f'Last login: {ll_str}<br>Joined: {joined_str}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout_user()
                st.rerun()
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Live input summary ────────────────────────────────────────────
        st.markdown('<div class="sidebar-title">📊 Live Input Summary</div>', unsafe_allow_html=True)
        rows = [
            ("City", city), ("Locality", loc), ("Area", f"{area:,} sqft"),
            ("Config", f"{bhk} BHK"), ("Baths", str(baths)),
            ("Age", f"{age} yrs"), ("Floor", f"{floor}/{t_fl}"),
            ("Furnishing", furn.replace("Furnished","Furn.")),
            ("Parking", "✅" if park else "❌"),
            ("Lift", "✅" if lift else "❌"),
            ("East Facing", "✅" if east else "❌"),
        ]
        html = ""
        for lbl, val in rows:
            html += f'<div class="sb-row"><span class="sb-label">{lbl}</span><span class="sb-value">{val}</span></div>'
        st.markdown(html, unsafe_allow_html=True)


def render_hero(metrics):
    name = metrics["best_model"].split()[0] if metrics else "XGBoost"
    r2   = f"{metrics['metrics'][metrics['best_model']]['R2']:.4f}" if metrics else "—"
    st.markdown(f"""
<div class="hero">
  <div class="hero-title">House Price<br>Estimator</div>
  <p class="hero-sub">AI-powered property valuation for Indian real estate markets.<br>Instant estimates using ML models trained on real housing data.</p>
  <div style="margin-top:16px; display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <span class="xgb-badge"><span class="badge-dot"></span>{name} &nbsp;·&nbsp; R² {r2}</span>
    <span style="font-size:0.72rem;color:#334155;font-family:'JetBrains Mono',monospace;">10 Cities &nbsp;·&nbsp; Live ML Model</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_india_map(selected_city, city_stats):
    lat = [CITY_COORDS[c][0] for c in CITIES]
    lon = [CITY_COORDS[c][1] for c in CITIES]
    prices = [city_stats.get(c,{}).get("median_price_per_sqft",6000) if city_stats else 6000 for c in CITIES]
    sizes  = [18 if c == selected_city else 10 for c in CITIES]
    colors = ["#f472b6" if c == selected_city else "#6366f1" for c in CITIES]

    fig = go.Figure(go.Scattergeo(
        lat=lat, lon=lon,
        text=[f"<b>{c}</b><br>₹{p:,.0f}/sqft" for c, p in zip(CITIES, prices)],
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color="white"),
        marker=dict(
            size=sizes, color=colors,
            line=dict(width=1.5, color="rgba(255,255,255,0.4)"),
            opacity=0.9,
        ),
        hoverinfo="text",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            scope="asia",
            center=dict(lat=22, lon=80),
            projection_scale=4.5,
            bgcolor="rgba(0,0,0,0)",
            showland=True, landcolor="rgba(15,23,42,0.7)",
            showocean=True, oceancolor="rgba(5,10,20,0.5)",
            showlakes=False,
            showcountries=True, countrycolor="rgba(99,102,241,0.3)",
            showcoastlines=True, coastlinecolor="rgba(99,102,241,0.2)",
            showframe=False,
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_empty_state():
    st.markdown("""
<div class="empty-card">
  <div class="empty-icon">🏠</div>
  <div class="empty-title">No estimate generated yet</div>
  <div class="empty-hint">Configure the property details and click Estimate Price.</div>
</div>
""", unsafe_allow_html=True)


def render_result(price: float, inputs: dict, city_stats):
    low  = price * 0.88
    high = price * 1.12
    ppsf = price / max(inputs["area_sqft"], 1)

    insight = ""
    if city_stats:
        overall  = city_stats.get("_overall_median", {}).get("median_price_per_sqft", 8000)
        city_med = city_stats.get(inputs["city"], {}).get("median_price_per_sqft", overall)
        ratio    = (price / inputs["area_sqft"]) / city_med if city_med else 1
        if ratio > 1.25:
            insight = f"This estimate is above the median for comparable properties in {inputs['city']}."
        elif ratio < 0.80:
            insight = f"Strong value — estimate is below the median rate for {inputs['city']}."
        else:
            insight = f"Aligned with current market rates for {inputs['locality_tier'].lower()} properties in {inputs['city']}."

    st.markdown(f"""
<div class="res-card">
  <div class="res-eyebrow">⬡ Estimated Market Value</div>
  <div class="res-price">{fmt_inr(price)}</div>
  <div class="res-context">
    {inputs['city']} · {inputs['locality_tier']} · {inputs['bhk']} BHK · {inputs['area_sqft']:,} sqft
  </div>
  <div class="range-row">
    <div class="range-cell">
      <div class="range-lbl">Low</div>
      <div class="range-val">{fmt_inr(low, compact=True)}</div>
    </div>
    <div class="range-cell">
      <div class="range-lbl">Estimate</div>
      <div class="range-val accent">{fmt_inr(price, compact=True)}</div>
    </div>
    <div class="range-cell">
      <div class="range-lbl">High</div>
      <div class="range-val">{fmt_inr(high, compact=True)}</div>
    </div>
    <div class="range-cell">
      <div class="range-lbl">Per sqft</div>
      <div class="range-val">{fmt_inr(ppsf, compact=True)}</div>
    </div>
  </div>
  <span class="conf-tag">Confidence interval ±12%</span>
  {"<div class='insight-wrap'>" + insight + "</div>" if insight else ""}
</div>
""", unsafe_allow_html=True)


def tab_city_prices(city_stats, selected_city):
    st.markdown('<div class="chart-title">Median Price per Square Foot by City</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Derived from training data. Selected city is highlighted.</div>', unsafe_allow_html=True)
    if not city_stats:
        st.caption("Run `python src/train.py` to generate city data.")
        return
    names = [c for c in city_stats if not c.startswith("_")]
    vals  = [city_stats[c]["median_price_per_sqft"] for c in names]
    clrs  = []
    glow_clrs = []
    for c in names:
        if c == selected_city:
            clrs.append("rgba(244,114,182,1)")
            glow_clrs.append("rgba(244,114,182,0.3)")
        else:
            clrs.append("rgba(99,102,241,0.7)")
            glow_clrs.append("rgba(99,102,241,0.1)")

    pairs = sorted(zip(vals, names, clrs), reverse=True)
    sv, sn, sc = zip(*pairs)

    fig = go.Figure(go.Bar(
        y=list(sn), x=list(sv), orientation="h",
        marker=dict(
            color=list(sc),
            line=dict(width=0),
        ),
        text=[fmt_inr(v, compact=True) for v in sv],
        textfont=dict(color="#e2e8f0", size=10, family="JetBrains Mono"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f} / sqft<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_DARK, height=320,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#94a3b8", family="Space Grotesk")),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def tab_value_drivers(inp):
    st.markdown('<div class="chart-title">Value Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Illustrative breakdown of how each factor contributes to the price estimate.</div>', unsafe_allow_html=True)
    if not inp:
        st.caption("Run an estimation first.")
        return
    base = CITY_BASE_PPSF.get(inp["city"], 8000)
    lm   = {"Premium":1.45,"Mid":1.00,"Budget":0.65}.get(inp["locality_tier"],1.0)
    bv   = base * lm * inp["area_sqft"]
    furn_delta = {"Unfurnished":0,"Semi-Furnished":bv*0.08,"Fully Furnished":bv*0.18}.get(inp["furnishing"],0)
    drivers = {
        "Base value":     bv,
        "Locality":       (lm-1)*base*inp["area_sqft"],
        "Parking":        inp["parking"]*base*30,
        "Lift":           inp["lift"]*50_000,
        "East facing":    inp["east_facing"]*base*15,
        "Depreciation":   -inp["property_age"]*base*0.8,
        "Floor premium":  (inp["floor"]/max(inp["total_floors"],1))*base*40,
        "Furnishing":     furn_delta,
    }
    labels = list(drivers.keys())
    values = list(drivers.values())
    colors = ["rgba(99,102,241,0.8)" if i==0 else ("rgba(52,211,153,0.8)" if v>=0 else "rgba(248,113,113,0.8)") for i,v in enumerate(values)]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[fmt_inr(abs(v),compact=True) for v in values],
        textfont=dict(size=9, color="#e2e8f0", family="JetBrains Mono"),
        textposition="outside",
        hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_DARK, height=300,
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)", zeroline=True,
                   zerolinecolor="rgba(99,102,241,0.3)"),
        xaxis=dict(tickfont=dict(size=10, color="#94a3b8", family="Space Grotesk"), showgrid=False),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# def tab_model_performance(metrics):
#     st.markdown('<div class="chart-title">Model Evaluation</div>', unsafe_allow_html=True)
#     st.markdown('<div class="chart-desc">Metrics evaluated on a held-out 20% test set not seen during training.</div>', unsafe_allow_html=True)
#     if not metrics:
#         st.caption("Run `python src/train.py` to generate metrics.")
#         return
#     best = metrics["best_model"]
#     bm   = metrics["metrics"][best]
#     c1, c2, c3 = st.columns(3)
#     for col, lbl, val, sub in [
#         (c1,"RMSE",fmt_inr(bm["RMSE"]),"Root mean square error"),
#         (c2,"MAE", fmt_inr(bm["MAE"]), "Mean absolute error"),
#         (c3,"R²",  f"{bm['R2']:.4f}",  "Coefficient of determination"),
#     ]:
#         col.markdown(
#             f"<div class='perf-tile'>"
#             f"<div class='perf-lbl'>{lbl}</div>"
#             f"<div class='perf-val'>{val}</div>"
#             f"<div class='perf-sub'>{sub}</div>"
#             f"</div>", unsafe_allow_html=True)
#     st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
#     rows = [{"Model":n,"RMSE":fmt_inr(m["RMSE"]),"MAE":fmt_inr(m["MAE"]),
#              "R²":f"{m['R2']:.4f}","Best":"★" if n==best else ""}
#             for n,m in metrics["metrics"].items()]
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
#     comp = ROOT/"outputs"/"model_comparison.png"
#     # if comp.exists():
#     #     # st.image(str(comp), use_container_width=True)
#     #     st.pyplot(comp)
#     if comp.exists():
#         st.image(str(comp), use_container_width=True)
#     else:
#         st.info("Run `python src/train.py` to generate the model comparison chart.")
def tab_model_performance(metrics):
    st.markdown('<div class="chart-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Metrics evaluated on a held-out 20% test set not seen during training.</div>', unsafe_allow_html=True)
    if not metrics:
        st.caption("Run `python src/train.py` to generate metrics.")
        return
    best = metrics["best_model"]
    bm   = metrics["metrics"][best]
    c1, c2, c3 = st.columns(3)
    for col, lbl, val, sub in [
        (c1,"RMSE",fmt_inr(bm["RMSE"]),"Root mean square error"),
        (c2,"MAE", fmt_inr(bm["MAE"]), "Mean absolute error"),
        (c3,"R²",  f"{bm['R2']:.4f}",  "Coefficient of determination"),
    ]:
        col.markdown(
            f"<div class='perf-tile'>"
            f"<div class='perf-lbl'>{lbl}</div>"
            f"<div class='perf-val'>{val}</div>"
            f"<div class='perf-sub'>{sub}</div>"
            f"</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    rows = [{"Model":n,"RMSE":fmt_inr(m["RMSE"]),"MAE":fmt_inr(m["MAE"]),
             "R²":f"{m['R2']:.4f}","Best":"★" if n==best else ""}
            for n,m in metrics["metrics"].items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Chart
    import numpy as np
    models = list(metrics["metrics"].keys())
    rmse_vals = [metrics["metrics"][m]["RMSE"] for m in models]
    mae_vals  = [metrics["metrics"][m]["MAE"]  for m in models]
    r2_vals   = [metrics["metrics"][m]["R2"]   for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, rmse_vals); axes[0].set_title("RMSE"); axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15)
    axes[1].bar(x, mae_vals);  axes[1].set_title("MAE");  axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15)
    axes[2].bar(x, r2_vals);   axes[2].set_title("R²");   axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def tab_market_explorer(df):
    st.markdown('<div class="chart-title">Price vs Area</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-desc">Explore how property price varies with different attributes. Sample of 2,500 listings.</div>', unsafe_allow_html=True)
    if df is None:
        st.caption("Run `python data/generate_data.py` to load market data.")
        return
    c1, c2 = st.columns(2)
    xaxis = c1.selectbox("X axis",  ["area_sqft","property_age","bhk","floor"], key="xax")
    color = c2.selectbox("Group by",["city","locality_tier","furnishing"],       key="col")
    sample = df.sample(min(2500,len(df)),random_state=42)
    fig1 = px.scatter(
        sample, x=xaxis, y="price", color=color,
        opacity=0.6, template="plotly_dark",
        color_discrete_sequence=["#6366f1","#f472b6","#34d399","#fbbf24","#60a5fa","#a78bfa","#fb7185","#38bdf8","#4ade80","#facc15"],
        hover_data=["city","bhk","area_sqft"],
    )
    fig1.update_layout(**PLOTLY_DARK, height=300,
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94a3b8")),
    )
    fig1.update_yaxes(tickprefix="₹")
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Price Distribution by City</div>', unsafe_allow_html=True)
    fig2 = px.box(df, x="city", y="price", color="locality_tier",
                  template="plotly_dark",
                  color_discrete_map={"Premium":"#f472b6","Mid":"#6366f1","Budget":"#38bdf8"})
    fig2.update_layout(**PLOTLY_DARK, height=300,
        xaxis=dict(tickangle=-20),
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94a3b8")),
    )
    fig2.update_yaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ── Dashboard ─────────────────────────────────────────────────────────────────

def render_dashboard(user_id: int):
    """Full dashboard: stats, history table with delete, charts."""
    import pandas as pd

    # ── Stats ──────────────────────────────────────────────────────────────
    with st.spinner("Loading your dashboard..."):
        stats = get_user_stats(user_id)
        preds = get_user_predictions(user_id)

    total     = stats["total"]
    avg_price = stats["avg_price"]

    # ── Metric tiles ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f'<div class="dash-metric">'
        f'<div class="dash-metric-lbl">Total Predictions</div>'
        f'<div class="dash-metric-val">{total}</div>'
        f'<div class="dash-metric-sub">All time</div>'
        f'</div>', unsafe_allow_html=True)
    m2.markdown(
        f'<div class="dash-metric">'
        f'<div class="dash-metric-lbl">Average Estimate</div>'
        f'<div class="dash-metric-val">{fmt_inr(avg_price, compact=True) if avg_price else "—"}</div>'
        f'<div class="dash-metric-sub">Across all properties</div>'
        f'</div>', unsafe_allow_html=True)
    cities_explored = len({p["city"] for p in preds}) if preds else 0
    m3.markdown(
        f'<div class="dash-metric">'
        f'<div class="dash-metric-lbl">Cities Explored</div>'
        f'<div class="dash-metric-val">{cities_explored}</div>'
        f'<div class="dash-metric-sub">Unique cities</div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    if not preds:
        st.markdown("""
        <div class="empty-card">
          <div class="empty-icon">📊</div>
          <div class="empty-title">No predictions yet</div>
          <div class="empty-hint">Run your first estimation to see your dashboard.</div>
        </div>""", unsafe_allow_html=True)
        return

    df_p = pd.DataFrame(preds)
    df_p["created_at"] = pd.to_datetime(df_p["created_at"])
    df_p["predicted_price"] = df_p["predicted_price"].astype(float)

    # ── Charts ──────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="chart-title">Prediction History</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Estimated prices over time</div>', unsafe_allow_html=True)
        df_sorted = df_p.sort_values("created_at")
        fig_line = go.Figure(go.Scatter(
            x=df_sorted["created_at"],
            y=df_sorted["predicted_price"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2),
            marker=dict(color="#a78bfa", size=7,
                        line=dict(color="#6366f1", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
        ))
        fig_line.update_layout(
            **PLOTLY_DARK, height=260,
            xaxis=dict(showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)",
                       tickprefix="₹", tickformat=",.0f",
                       tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_line, use_container_width=True,
                        config={"displayModeBar": False})

    with ch2:
        st.markdown('<div class="chart-title">Predictions by City</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-desc">Number of estimates per city</div>', unsafe_allow_html=True)
        city_counts = df_p["city"].value_counts().reset_index()
        city_counts.columns = ["city", "count"]
        fig_bar = go.Figure(go.Bar(
            x=city_counts["city"],
            y=city_counts["count"],
            marker=dict(
                color=["rgba(244,114,182,0.85)" if i == 0 else "rgba(99,102,241,0.7)"
                       for i in range(len(city_counts))],
                line=dict(width=0),
            ),
            text=city_counts["count"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=10),
            hovertemplate="%{x}: %{y} predictions<extra></extra>",
        ))
        fig_bar.update_layout(
            **PLOTLY_DARK, height=260,
            xaxis=dict(showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)",
                       tickfont=dict(size=9)),
            bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True,
                        config={"displayModeBar": False})

    # ── History table with delete ────────────────────────────────────────────
    st.markdown('<div class="sec-head">🗂️ Prediction History</div>', unsafe_allow_html=True)

    for i, row in enumerate(preds):
        ts = row["created_at"]
        date_str = ts.strftime("%d %b %Y, %H:%M") if hasattr(ts, "strftime") else str(ts)
        price_str = fmt_inr(float(row["predicted_price"]))
        col_info, col_del = st.columns([8, 1])
        with col_info:
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.5);border:1px solid rgba(99,102,241,0.18);
                            border-radius:10px;padding:10px 14px;margin-bottom:6px;
                            display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;color:#e2e8f0;">
                        🏙️ <b>{row['city']}</b> &nbsp;·&nbsp;
                        {row['locality']} &nbsp;·&nbsp;
                        {row['area_sqft']:,} sqft &nbsp;·&nbsp;
                        {row['bhk']} BHK
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                                 color:#a78bfa;font-weight:600;">{price_str}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#475569;">
                        {date_str}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_del:
            if st.button("🗑️", key=f"del_{row['id']}_{i}",
                         help="Delete this prediction"):
                with st.spinner("Deleting..."):
                    ok = delete_prediction(row["id"], user_id)
                if ok:
                    st.success("Prediction deleted.", icon="✅")
                    st.rerun()
                else:
                    st.error("Could not delete prediction.")


# ── Profile ───────────────────────────────────────────────────────────────────

def render_profile(user_id: int):
    """Show user profile: avatar, username, email, join date, total predictions."""
    with st.spinner("Loading profile..."):
        profile = get_user_profile(user_id)

    if not profile:
        st.error("Could not load profile information.")
        return

    username   = profile.get("username", "—")
    email      = profile.get("email", "—")
    created_at = profile.get("created_at")
    total_pred = int(profile.get("total_predictions", 0))
    joined_str = created_at.strftime("%d %B %Y") if created_at and hasattr(created_at, "strftime") else "—"

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown(
            f"""
            <div class="profile-card">
              <div class="profile-avatar">👤</div>
              <div class="profile-name">{username}</div>
              <div class="profile-email">{email}</div>
              <div class="profile-stat">
                <span class="profile-stat-lbl">Username</span>
                <span class="profile-stat-val">{username}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-lbl">Email</span>
                <span class="profile-stat-val">{email}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-lbl">Member Since</span>
                <span class="profile-stat-val">{joined_str}</span>
              </div>
              <div class="profile-stat" style="border-bottom:none;">
                <span class="profile-stat-lbl">Total Predictions</span>
                <span class="profile-stat-val">{total_pred}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_auth_page():
    """Render the login / register landing page."""
    inject_css()

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-card">'
        '<div class="auth-logo">🏠</div>'
        '<div class="auth-title">House Price Estimator</div>'
        '<div class="auth-sub">AI-powered property valuation · Sign in to continue</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render inside an actual container so Streamlit widgets work
    with st.container():
        st.markdown(
            '<div style="background:rgba(15,23,42,0.75);backdrop-filter:blur(28px);'
            'border:1px solid rgba(99,102,241,0.3);border-radius:20px;'
            'padding:36px;margin-top:-8px;position:relative;">',
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["🔑  Login", "✨  Register"])

        # ── Login ───────────────────────────────────────────────────────────
        with login_tab:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            l_email = st.text_input("Email", key="l_email", placeholder="you@example.com")
            l_pass  = st.text_input("Password", type="password", key="l_pass", placeholder="••••••••")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Sign In", key="login_btn", use_container_width=True):
                if not l_email or not l_pass:
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Authenticating..."):
                        ok, result = login_user(l_email, l_pass)
                    if ok:
                        _set_session(result)
                        st.rerun()
                    else:
                        st.error(result)

        # ── Register ────────────────────────────────────────────────────────
        with register_tab:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            r_user    = st.text_input("Username",         key="r_user",    placeholder="johndoe")
            r_email   = st.text_input("Email",            key="r_email",   placeholder="you@example.com")
            r_pass    = st.text_input("Password",         type="password", key="r_pass",  placeholder="Min 8 chars, letters + digits")
            r_confirm = st.text_input("Confirm Password", type="password", key="r_confirm", placeholder="Repeat password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Create Account", key="register_btn", use_container_width=True):
                with st.spinner("Creating account..."):
                    ok, msg = register_user(r_user, r_email, r_pass, r_confirm)
                if ok:
                    st.success("✅ Account created! Switch to Login to sign in.")
                else:
                    st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    # Initialise DB (creates table if needed) — runs once per process
    try:
        init_db()
    except Exception as e:
        st.error(f"⚠️ Database connection failed: {e}")
        st.stop()

    # ── Auth gate ──────────────────────────────────────────────────────────
    if not is_authenticated():
        render_auth_page()
        st.stop()

    inject_css()

    model      = load_model()
    metrics    = load_metrics()
    city_stats = load_city_stats()
    df_market  = load_market_data()

    render_hero(metrics)

    if model is None:
        st.markdown("""
<div class="glass glow-card" style="text-align:center;padding:56px;max-width:500px;margin:60px auto;">
  <div style="font-size:2.5rem;margin-bottom:16px;">⚠️</div>
  <p style="color:#e2e8f0;font-size:0.95rem;font-weight:600;margin-bottom:8px;font-family:'Space Grotesk',sans-serif;">No trained model found</p>
  <p style="color:#475569;font-size:0.825rem;">Run <code style="background:rgba(99,102,241,0.15);padding:3px 10px;border-radius:6px;color:#a78bfa;font-family:'JetBrains Mono',monospace;">python src/train.py</code> to train a model.</p>
</div>""", unsafe_allow_html=True)
        st.stop()

    # ── Two-column layout ──────────────────────────────────────────────────────
    col_in, col_out = st.columns([5, 4], gap="large")

    with col_in:
        st.markdown('<div class="sec-head">📍 Location</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", CITIES, index=0, key="city",
                             help="Select the city where the property is located")
        loc  = c2.selectbox("Locality Tier", LOCALITY_OPTS, index=1, key="loc",
                             help="Premium: prime area · Mid: suburban · Budget: outskirts")

        st.markdown('<div class="sec-head">🏗️ Property Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        area  = c1.number_input("Carpet Area (sqft)", 250, 5000, 1000, step=50, key="area",
                                 help="Total carpet area in square feet")
        bhk   = c2.selectbox("BHK Configuration", [1,2,3,4,5], index=1, key="bhk",
                              help="Number of bedrooms, hall and kitchen")
        baths = c3.selectbox("Bathrooms", [1,2,3,4,5], index=1, key="baths",
                              help="Number of bathrooms/toilets")

        st.markdown('<div class="sec-head">🏢 Building Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age   = c1.number_input("Property Age (yrs)", 0, 40, 5, key="age",
                                  help="Age of the property in years")
        t_fl  = c2.number_input("Total Floors", 2, 40, 10, key="tfl",
                                  help="Total number of floors in the building")
        floor = c3.number_input("Unit Floor", 0, int(t_fl), min(5,int(t_fl)), key="fl",
                                 help="Floor number of your unit")

        st.markdown('<div class="sec-head">✨ Additional Features</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3,2])
        furn = c1.selectbox("Furnishing Status", FURNISHING_OPTS, index=1, key="furn",
                             help="Current furnishing level of the property")
        with c2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            park = st.toggle("🚗 Covered Parking", value=True,  key="park")
            lift = st.toggle("🛗 Lift / Elevator",  value=True,  key="lift")
            east = st.toggle("🧭 East Facing",      value=False, key="east")

        render_sidebar(city, loc, area, bhk, baths, age, t_fl, floor, furn, park, lift, east)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        clicked = st.button("⚡ Estimate Price", key="cta")

    with col_out:
        if clicked:
            # Input validation
            if area < 100:
                st.error("⚠️ Area must be at least 100 sqft.")
            elif bhk < 1 or bhk > 10:
                st.error("⚠️ BHK must be between 1 and 10.")
            else:
                inputs = {
                    "city": city, "locality_tier": loc,
                    "area_sqft": area, "bhk": bhk, "bathrooms": baths,
                    "floor": floor, "total_floors": t_fl,
                    "parking": int(park), "lift": int(lift),
                    "east_facing": int(east), "furnishing": furn,
                    "property_age": age,
                }
                with st.spinner("🔮 Calculating estimate..."):
                    time.sleep(0.35)
                    try:
                        price = run_prediction(model, inputs)
                    except Exception as e:
                        price = None
                        st.error(f"⚠️ Prediction failed: {e}")
                if price:
                    st.session_state["price"]  = price
                    st.session_state["inputs"] = inputs
                    st.success("✅ Estimate ready")
                    # Auto-save prediction
                    user = get_current_user()
                    if user and user.get("user_id"):
                        with st.spinner("Saving prediction..."):
                            save_prediction(
                                user_id       = user["user_id"],
                                city          = city,
                                locality      = loc,
                                area_sqft     = int(area),
                                bhk           = int(bhk),
                                predicted_price = price,
                            )
                elif price is not None:
                    st.error("⚠️ Prediction returned an unexpected value.")

        p  = st.session_state.get("price")
        si = st.session_state.get("inputs", {})
        if p:
            render_result(p, si, city_stats)
        else:
            render_empty_state()

    # ── India Map ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("🗺️ City Hotspot Map — India", expanded=False):
        render_india_map(city, city_stats)

    # ── Analytics & Feature Tabs ────────────────────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,102,241,0.15);margin:32px 0;'>", unsafe_allow_html=True)

    active_city = st.session_state.get("inputs", {}).get("city", city)
    active_inp  = st.session_state.get("inputs", {})
    user        = get_current_user()
    user_id     = user.get("user_id") if user else None

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Dashboard",
        "👤 Profile",
        "🏙️ City Prices",
        "📈 Value Drivers",
        "🤖 Model Performance",
        "🗺️ Market Data",
    ])
    with t1:
        if user_id:
            render_dashboard(user_id)
        else:
            st.warning("Please log in to view your dashboard.")
    with t2:
        if user_id:
            render_profile(user_id)
        else:
            st.warning("Please log in to view your profile.")
    with t3: tab_city_prices(city_stats, active_city)
    with t4: tab_value_drivers(active_inp)
    with t5: tab_model_performance(metrics)
    with t6: tab_market_explorer(df_market)

    st.markdown(
        "<p style='text-align:center;font-size:0.68rem;color:#1e293b;margin-top:40px;"
        "font-family:JetBrains Mono,monospace;'>India House Price Estimator &nbsp;·&nbsp; scikit-learn + XGBoost</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
