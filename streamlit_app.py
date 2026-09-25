"""
Streamlit UI for the Mental Health Score predictor.

Runs in "direct mode" by default — it imports src.inference and calls the
same model-loading/prediction code the FastAPI app and SageMaker handler
use, so you don't need the API server running separately. Every prediction
is also logged through src.monitoring, exactly like a real /predict call,
so the Live Monitoring tab reflects genuine drift stats.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, monitoring
from src.inference import ModelNotLoadedError, load_model, predict
from src.schema import StudentFeatures

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CSS — animated gradient background, glass cards, fade-ins, pulsing button
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a3d);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.6s ease-out both; }

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7ee8fa, #eec0c6, #7ee8fa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
    margin-bottom: 0;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.hero-sub { color: #b7b7d9; font-size: 1.02rem; margin-top: -6px; }

.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
.badge-ok    { background: rgba(52, 211, 153, 0.18); color: #34d399; border: 1px solid rgba(52,211,153,.4);}
.badge-warn  { background: rgba(251, 191, 36, 0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,.4);}
.badge-alert { background: rgba(248, 113, 113, 0.18); color: #f87171; border: 1px solid rgba(248,113,113,.4);}
.badge-nodata{ background: rgba(148, 163, 184, 0.18); color: #94a3b8; border: 1px solid rgba(148,163,184,.4);}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    transition: all 0.25s ease;
    box-shadow: 0 0 0 rgba(124, 58, 237, 0.0);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.55);
}
div.stButton > button:active { transform: translateY(0px) scale(0.98); }

[data-testid="stMetricValue"] { font-size: 2.1rem; }
</style>
""", unsafe_allow_html=True)

FEATURE_KEYS = list(StudentFeatures.model_fields.keys())

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, score, inputs

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model status")
    try:
        load_model()
        st.markdown('<span class="badge badge-ok">● model loaded</span>', unsafe_allow_html=True)
        st.caption(f"`{config.MODEL_PATH.name}` · scikit-learn 1.6.1 pinned")
    except ModelNotLoadedError as exc:
        st.markdown('<span class="badge badge-alert">● model failed to load</span>', unsafe_allow_html=True)
        st.error(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🧭 About")
    st.caption(
        "Predicts a 0–10 mental health score from a student's lifestyle "
        "and social-media-usage features. This is a technical demo, "
        "**not** a diagnostic or clinical tool."
    )
    st.divider()
    if st.button("🗑️ Clear session history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="fade-in"><p class="hero-title">🧠 Mental Health Score Predictor</p>'
    '<p class="hero-sub">Lifestyle & social-media inputs → predicted well-being score (0–10)</p></div>',
    unsafe_allow_html=True,
)
st.write("")

tab_predict, tab_history, tab_monitor = st.tabs(["🎯 Predict", "📈 Session history", "🩺 Live monitoring"])

# ==========================================================================
# TAB 1 — Predict
# ==========================================================================
with tab_predict:
    col_form, col_result = st.columns([1, 1.15], gap="large")

    with col_form:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("#### Student profile")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 10, 100, 21)
            study_hours = st.slider("Study hours / day", 0.0, 15.0, 4.5, 0.1)
            usage_hours = st.slider("Social media use (hrs/day)", 0.0, 24.0, 4.0, 0.1)
            unlocks = st.slider("Phone unlocks / day", 0, 300, 134)
        with c2:
            activity_hours = st.slider("Physical activity (hrs/day)", 0.0, 24.0, 2.2, 0.1)
            sleep_hours = st.slider("Sleep (hrs/night)", 0.0, 24.0, 6.7, 0.1)
            stress = st.select_slider("Stress level", options=config.STRESS_LEVEL_ORDER, value="Medium")
            gender = st.selectbox("Gender", config.GENDER_VALUES)

        c3, c4 = st.columns(2)
        with c3:
            academic = st.selectbox("Academic level", config.ACADEMIC_LEVEL_VALUES)
            platform = st.selectbox("Most-used platform", config.PLATFORM_VALUES, index=0)
        with c4:
            purpose = st.selectbox("Primary purpose of use", config.PURPOSE_VALUES)
            country = st.text_input("Country", value="India")

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        predict_clicked = st.button("✨ Predict my score", use_container_width=True)

    with col_result:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        gauge_slot = st.empty()
        message_slot = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        def render_gauge(value: float, max_val: float = 10.0):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": " / 10", "font": {"size": 42, "color": "#f5f5fa"}},
                gauge={
                    "axis": {"range": [0, max_val], "tickcolor": "#8888aa"},
                    "bar": {"color": "#a78bfa", "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 4], "color": "rgba(248,113,113,0.35)"},
                        {"range": [4, 6], "color": "rgba(251,191,36,0.35)"},
                        {"range": [6, 8], "color": "rgba(163,230,53,0.35)"},
                        {"range": [8, 10], "color": "rgba(52,211,153,0.35)"},
                    ],
                },
            ))
            fig.update_layout(
                height=300, margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e5f5"},
            )
            return fig

        if not predict_clicked and not st.session_state.history:
            gauge_slot.plotly_chart(render_gauge(0), use_container_width=True, key="gauge_idle")
            message_slot.info("Set the sliders and click **Predict my score** to run the model.")

        elif not predict_clicked and st.session_state.history:
            last = st.session_state.history[-1]
            gauge_slot.plotly_chart(render_gauge(last["score"]), use_container_width=True, key="gauge_last")
            message_slot.caption(f"Showing your last prediction from {last['timestamp']}.")

        if predict_clicked:
            try:
                features = StudentFeatures(
                    Study_Hours=study_hours, Age=age, Avg_Daily_Usage_Hours=usage_hours,
                    Daily_Unlocks=unlocks, Physical_Activity_Hours=activity_hours,
                    Sleep_Hours_Per_Night=sleep_hours, Stress_Level=stress, Gender=gender,
                    Academic_Level=academic, Most_Used_Platform=platform,
                    Purpose_Of_Use=purpose, Country=country,
                )
            except Exception as exc:  # pydantic ValidationError
                st.error(f"Invalid input: {exc}")
                st.stop()

            row = features.to_model_row()

            with st.spinner("Running inference..."):
                time.sleep(0.35)  # lets the spinner + card feel intentional, not instant
                try:
                    [score] = predict([row])
                except ModelNotLoadedError as exc:
                    st.error(str(exc))
                    st.stop()

            # ---- rising-gauge animation: sweep from 0 up to the real score ----
            for v in np.linspace(0, score, 16):
                gauge_slot.plotly_chart(render_gauge(round(v, 2)), use_container_width=True, key=f"gauge_{v}")
                time.sleep(0.02)
            gauge_slot.plotly_chart(render_gauge(score), use_container_width=True, key="gauge_final")

            if score >= 8:
                tier, tone = "Excellent", "success"
            elif score >= 6:
                tier, tone = "Good", "success"
            elif score >= 4:
                tier, tone = "Moderate", "warning"
            else:
                tier, tone = "Needs attention", "error"
            getattr(message_slot, tone)(f"**{tier}** — predicted score **{score:.2f} / 10**")

            if score >= 8:
                st.balloons()

            monitoring.log_prediction(row, score, model_version="streamlit-direct")
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "score": score,
                **row,
            })

# ==========================================================================
# TAB 2 — Session history
# ==========================================================================
with tab_history:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No predictions yet this session — run one from the **Predict** tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions this session", len(hist_df))
        c2.metric("Average score", f"{hist_df['score'].mean():.2f}")
        c3.metric("Latest score", f"{hist_df['score'].iloc[-1]:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist_df) + 1)), y=hist_df["score"],
            mode="lines+markers", line=dict(color="#a78bfa", width=3, shape="spline"),
            marker=dict(size=8, color="#db2777"), fill="tozeroy",
            fillcolor="rgba(167,139,250,0.15)",
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e5f5"}, xaxis_title="Prediction #", yaxis_title="Score",
            yaxis=dict(range=[0, 10], gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hist_df[::-1], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# TAB 3 — Live monitoring (drift against the training baseline)
# ==========================================================================
with tab_monitor:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("#### Drift vs. training baseline")
    st.caption(
        "Compares predictions logged this session (and any prior local runs) "
        "against `monitoring/reference_stats.json` — PSI for categoricals, "
        "standardized mean-shift for numerics."
    )
    window = st.slider("Window size (most recent predictions)", 10, 1000, 200, key="drift_window")
    refresh = st.button("🔄 Refresh drift report")

    try:
        report = monitoring.compute_drift_report(window=window)
    except FileNotFoundError as exc:
        report = None
        st.error(str(exc))

    if report:
        status = report["status"]
        badge_class = {"ok": "badge-ok", "warn": "badge-warn", "alert": "badge-alert", "no_data": "badge-nodata"}[status]
        st.markdown(
            f'Overall status: <span class="badge {badge_class}">{status.upper()}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;window: {report["window_size"]} rows '
            f'(reference: {report["reference_size"]} rows)',
            unsafe_allow_html=True,
        )
        st.write("")

        if report["status"] == "no_data":
            st.info("No predictions logged yet — run some from the **Predict** tab first.")
        else:
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Numeric feature drift** (z-score of mean shift)")
                for feat, d in report["numeric_drift"].items():
                    b = {"ok": "badge-ok", "warn": "badge-warn", "alert": "badge-alert"}[d["severity"]]
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                        f'<span>{feat}</span>'
                        f'<span class="badge {b}">z={d["drift_zscore"]}</span></div>',
                        unsafe_allow_html=True,
                    )
            with colB:
                st.markdown("**Categorical feature drift** (PSI)")
                for feat, d in report["categorical_drift"].items():
                    b = {"ok": "badge-ok", "warn": "badge-warn", "alert": "badge-alert"}[d["severity"]]
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                        f'<span>{feat}</span>'
                        f'<span class="badge {b}">PSI={d["psi"]}</span></div>',
                        unsafe_allow_html=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)
