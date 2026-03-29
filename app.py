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
    page_icon="",
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
    background: #060d0b !important;
    color: #ffffff;
}
.stApp {
    background: #060d0b !important;
    min-height: 100vh;
}
.main .block-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0px 32px 80px 32px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Top Navbar ── */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 0;
    border-bottom: 1px solid rgba(0, 255, 136, 0.2);
    margin-bottom: 40px;
}
.nav-left {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.nav-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #888;
}
.nav-btn {
    background: transparent;
    border: 1px solid #00ff88;
    color: #00ff88;
    padding: 6px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
}
.nav-btn:hover {
    background: rgba(0, 255, 136, 0.1);
}

/* ── Hero Section ── */
.hero-section {
    text-align: center;
    margin-bottom: 40px;
}
.hero-title {
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #ffffff;
    margin-bottom: 12px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.1rem;
    color: #888;
}

/* ── Pipeline Tabs ── */
.pipeline-wrap {
    margin-bottom: 60px;
}
.pipeline-tabs {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 16px;
}
.p-tab {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    padding: 8px 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: #0a1411;
    border-radius: 4px;
}
.p-tab.active {
    color: #00ff88;
    border-color: #00ff88;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
}
.p-bar-track {
    width: 100%;
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
}
.p-bar-fill {
    width: 80%; /* 4th tab is active */
    height: 100%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
}

/* ── Form Inputs ── */
.sec-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 12px;
    margin-top: 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 8px;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #0a1411 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    height: 48px !important;
    padding: 0 14px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 1px #00ff88 !important;
    outline: none !important;
}
div[data-testid="stNumberInput"] button {
    background: #0a1411 !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: #888 !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #00ff88 !important;
}
label {
    color: #aaa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stToggle"] label { color: #aaa !important; }
div[data-testid="stToggle"] input[type="checkbox"] { accent-color: #00ff88; }
div[data-testid="stToggle"] > label > div:first-child {
    background: #0a1411 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: transparent !important;
    color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0, 255, 136, 0.1) !important;
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.2) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #022b1c 0%, #061710 100%) !important;
    border-right: 1px solid rgba(0, 255, 136, 0.2) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}
section[data-testid="stSidebar"] * {
    color: #e0f2ec !important;
}
.sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #00ff88 !important;
    text-transform: uppercase;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0, 255, 136, 0.2);
}
.sb-user-card {
    background: rgba(0, 255, 136, 0.05);
    border: 1px solid rgba(0, 255, 136, 0.2);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}
.sb-user-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff !important;
    margin-bottom: 8px;
}
.sb-user-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #88c0a8 !important;
    line-height: 1.5;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0, 255, 136, 0.1);
}
.sb-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #88c0a8 !important;
}
.sb-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #00ff88 !important;
}

/* ── Panels ── */
.left-panel {
    background: #0a1411;
    border: 1px solid rgba(0, 255, 136, 0.3);
    padding: 32px;
    border-radius: 8px;
    position: relative;
}
/* Left panel watermark */
.watermark {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.02);
    pointer-events: none;
    z-index: 0;
    letter-spacing: 0.2em;
}

.right-panel {
    background: #0a1411;
    border: 1px solid rgba(0, 255, 136, 0.3);
    padding: 32px;
    border-radius: 8px;
    min-height: 100%;
}
.res-card {
    position: relative;
    z-index: 1;
}
.res-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
    margin-bottom: 24px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.res-price {
    font-size: 4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
}
.res-context {
    font-family: 'JetBrains Mono', monospace;
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 32px;
}
.conf-tag {
    display: inline-block;
    padding: 6px 12px;
    border: 1px solid #00ff88;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    border-radius: 4px;
    margin-bottom: 32px;
}
.range-row {
    display: flex;
    gap: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 24px;
}
.range-cell {
    flex: 1;
}
.range-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.range-val {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    color: #fff;
    font-weight: 500;
}

.empty-state {
    text-align: center;
    padding: 60px 0;
    font-family: 'JetBrains Mono', monospace;
    color: #666;
}

/* Auth Page overrides */
.auth-bg { display: none; }
.auth-card {
    background: #0a1411;
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 8px;
    box-shadow: none;
}
.auth-title { background: none; -webkit-text-fill-color: #fff; color: #fff; }
.auth-logo { border-color: #00ff88; background: transparent; box-shadow: none; color: #00ff88; }
.auth-divider { display: none; }

</style>
"""


PARTICLE_JS = ""


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
                f'<div class="sb-user-name">{user["username"]}</div>'
                f'<div class="sb-user-meta">'
                f'Last login: {ll_str}<br>Joined: {joined_str}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="logout_btn", use_container_width=True):
                logout_user()
                st.rerun()
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Live input summary ────────────────────────────────────────────
        st.markdown('<div class="sidebar-title">Live Input Summary</div>', unsafe_allow_html=True)
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
    st.markdown("""
<div class="top-nav" style="border-bottom: none; margin-bottom: 0px;">
  <div class="nav-left">House.Predict</div>
</div>

<div class="hero-section">
  <div class="hero-title">THE PREDICTION ENGINE</div>
  <div class="hero-sub">From raw inputs to accurate valuations in &lt; 500ms</div>
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
<div class="right-panel">
  <div class="empty-state">
    [ WAITING FOR INPUT ]<br><br>
    Configure property details on the left and initialize prediction sequence.
  </div>
</div>
""", unsafe_allow_html=True)



def render_result(price: float, inputs: dict, city_stats):
    low  = price * 0.88
    high = price * 1.12
    ppsf = price / max(inputs["area_sqft"], 1)

    st.markdown(f"""
<div class="right-panel">
  <div class="res-card">
    <div class="res-eyebrow">Prediction Output</div>
    <div class="res-price">{fmt_inr(price)}</div>
    <div class="res-context">
      {inputs['city'].upper()} // {inputs['locality_tier'].upper()} // {inputs['bhk']} BHK // {inputs['area_sqft']:,} SQFT
    </div>
    <div class="conf-tag">CONFIDENCE: ±12%</div>
    <div class="range-row">
      <div class="range-cell">
        <div class="range-lbl">Est. Low</div>
        <div class="range-val">{fmt_inr(low, compact=True)}</div>
      </div>
      <div class="range-cell">
        <div class="range-lbl">Est. High</div>
        <div class="range-val">{fmt_inr(high, compact=True)}</div>
      </div>
      <div class="range-cell">
        <div class="range-lbl">Price / Sqft</div>
        <div class="range-val">{fmt_inr(ppsf, compact=True)}</div>
      </div>
    </div>
  </div>
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
    st.markdown('<div class="sec-head">Prediction History</div>', unsafe_allow_html=True)

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
                        <b>{row['city']}</b> &nbsp;·&nbsp;
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
            if st.button("Delete", key=f"del_{row['id']}_{i}",
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
              <div class="profile-avatar"></div>
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

    # Gradient mesh background overlay (fixed, full-screen)
    st.markdown('<div class="auth-bg"></div>', unsafe_allow_html=True)

    # Center content using columns
    _, mid, _ = st.columns([1, 1.5, 1])

    with mid:
        # ── Header card (pure HTML) ──────────────────────────────────────
        st.markdown(
            '<div class="auth-card">'
            '<div class="auth-logo"></div>'
            '<div class="auth-title">House Price Estimator</div>'
            '<div class="auth-sub">AI-powered property valuation &nbsp;·&nbsp; India</div>'
            '<div class="auth-divider"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["🔑  Sign In", " Create Account"])

        # ── Login ───────────────────────────────────────────────────────────
        with login_tab:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            l_email = st.text_input("Email address", key="l_email", placeholder="you@example.com")
            l_pass  = st.text_input("Password", type="password", key="l_pass", placeholder="••••••••")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Sign In  →", key="login_btn", use_container_width=True):
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
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            r_user    = st.text_input("Username",         key="r_user",    placeholder="johndoe")
            r_email   = st.text_input("Email address",    key="r_email",   placeholder="you@example.com")
            r_pass    = st.text_input("Password",         type="password", key="r_pass",  placeholder="Min 8 chars, letters + digits")
            r_confirm = st.text_input("Confirm Password", type="password", key="r_confirm", placeholder="Repeat password")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Create Account  →", key="register_btn", use_container_width=True):
                with st.spinner("Creating account..."):
                    ok, msg = register_user(r_user, r_email, r_pass, r_confirm)
                if ok:
                    st.success("✅ Account created! Switch to Sign In to continue.")
                else:
                    st.error(msg)

        # ── Footer ──────────────────────────────────────────────────────────
        st.markdown(
            '<div class="auth-footer">'
            'India House Price Estimator &nbsp;·&nbsp; scikit-learn + XGBoost'
            '</div>',
            unsafe_allow_html=True,
        )


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
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="left-panel"><div class="watermark">PROCESSING</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Location</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", CITIES, index=0, key="city",
                             help="Select the city where the property is located")
        loc  = c2.selectbox("Locality Tier", LOCALITY_OPTS, index=1, key="loc",
                             help="Premium: prime area · Mid: suburban · Budget: outskirts")

        st.markdown('<div class="sec-head">Property Details</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="sec-head">Additional Features</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3,2])
        furn = c1.selectbox("Furnishing Status", FURNISHING_OPTS, index=1, key="furn",
                             help="Current furnishing level of the property")
        with c2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            park = st.toggle("Covered Parking", value=True,  key="park")
            lift = st.toggle("Lift / Elevator",  value=True,  key="lift")
            east = st.toggle("East Facing",      value=False, key="east")

        render_sidebar(city, loc, area, bhk, baths, age, t_fl, floor, furn, park, lift, east)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        clicked = st.button("INITIALIZE SEQUENCE", key="cta")
        st.markdown("</div>", unsafe_allow_html=True)

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
                with st.spinner("Calculating estimate..."):
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
    with st.expander("City Hotspot Map — India", expanded=False):
        render_india_map(city, city_stats)

    # ── Analytics & Feature Tabs ────────────────────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,102,241,0.15);margin:32px 0;'>", unsafe_allow_html=True)

    active_city = st.session_state.get("inputs", {}).get("city", city)
    active_inp  = st.session_state.get("inputs", {})
    user        = get_current_user()
    user_id     = user.get("user_id") if user else None

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Dashboard",
        "Profile",
        "City Prices",
        "Value Drivers",
        "Model Performance",
        "Market Data",
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
