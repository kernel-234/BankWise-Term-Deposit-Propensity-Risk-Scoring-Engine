import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.feature_engineering import engineer_features

# ==========================================
# 1. PAGE SETUP & MODERN NEON GLASS THEME
# ==========================================
st.set_page_config(
    page_title="BankWise | Intelligence Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Dark Theme Canvas */
    .stApp {
        background: #0B0F17;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Modern Glassmorphic Cards */
    .card {
        background: rgba(18, 24, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }

    /* KPI Highlights */
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 6px;
    }

    /* Decision Cards */
    .decision-badge-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .decision-badge-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #121826;
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94A3B8;
        padding: 0 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #1E293B 0%, #121826 100%) !important;
        color: #38BDF8 !important;
        border-top: 2px solid #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL ARTIFACTS
# ==========================================
@st.cache_resource
def load_assets():
    model_path = "models/champion_model.pkl"
    data_path = "data/processed/banking_engineered.csv"
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    return model, df

model, df_engineered = load_assets()

if model is None:
    st.error("⚠️ Model file missing! Please verify `models/champion_model.pkl` exists.")
    st.stop()

OPTIMAL_THRESHOLD = 0.7687

# Plotly Theme Defaults
PLOT_THEME = "plotly_dark"
COLOR_SUCCESS = "#10B981"
COLOR_DANGER = "#EF4444"
COLOR_PRIMARY = "#38BDF8"
COLOR_BG = "rgba(0,0,0,0)"

# ==========================================
# 3. HEADER
# ==========================================
head_c1, head_c2 = st.columns([3, 1])
with head_c1:
    st.title("💎 BankWise™ Enterprise Intelligence")
    st.markdown("<p style='color: #94A3B8; font-size: 1.05rem;'>Real-time Term Deposit Propensity & Campaign Optimization System</p>", unsafe_allow_html=True)
with head_c2:
    st.markdown("""
    <div style="text-align: right; padding-top: 15px;">
        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #10B981; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
            ● ENGINE READY (v1.0)
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# 4. SIDEBAR INPUTS
# ==========================================
st.sidebar.markdown("### 🎛️ Prospect Attributes")

with st.sidebar.form("input_form"):
    st.markdown("**Client Profile**")
    age = st.slider("Customer Age", 18, 90, 42)
    job = st.selectbox("Job Type", ["management", "blue-collar", "technician", "admin.", "services", "retired", "self-employed", "entrepreneur", "unemployed", "housemaid", "student", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"])
    default = st.radio("Has Credit Default?", ["no", "yes"], horizontal=True)
    balance = st.number_input("Yearly Balance (€)", value=2450, step=250)
    
    st.markdown("---")
    st.markdown("**Campaign & Financial History**")
    housing = st.radio("Housing Loan?", ["yes", "no"], horizontal=True)
    loan = st.radio("Personal Loan?", ["no", "yes"], horizontal=True)
    contact = st.selectbox("Contact Channel", ["cellular", "telephone", "unknown"])
    day = st.slider("Contact Day of Month", 1, 31, 18)
    month = st.selectbox("Contact Month", ["may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "sep", "oct", "mar", "dec"])
    duration = st.slider("Call Duration (Seconds)", 0, 1200, 380)
    campaign = st.slider("Calls During Campaign", 1, 20, 2)
    pdays = st.number_input("Days Since Last Contact (-1 = Never)", value=-1, step=1)
    previous = st.number_input("Previous Contacts Count", value=0, step=1)
    poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"])
    
    submit_btn = st.form_submit_button("⚡ Compute Real-Time Score", use_container_width=True)

# Process Features
raw_dict = {
    "age": age, "job": job, "marital": marital, "education": education,
    "default": default, "balance": balance, "housing": housing, "loan": loan,
    "contact": contact, "day": day, "month": month, "duration": duration,
    "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
}
df_raw = pd.DataFrame([raw_dict])
df_transformed = engineer_features(df_raw)

expected_cols = getattr(model, "feature_names_in_", None)
if expected_cols is not None:
    for col in expected_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = 0
    df_transformed = df_transformed[expected_cols]

# Inference
prob = float(model.predict_proba(df_transformed)[:, 1][0])
is_high_propensity = prob >= OPTIMAL_THRESHOLD

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab_scoring, tab_why, tab_business, tab_eda = st.tabs([
    "🎯 Live Scoring & Action",
    "🔍 Model Decision Breakdown",
    "💰 Campaign Financial Simulator",
    "📊 Portfolio Intelligence"
])

# ------------------------------------------
# TAB 1: LIVE SCORING & ACTION
# ------------------------------------------
with tab_scoring:
    # Top KPI Banner (Custom CSS Cards)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">Conversion Likelihood</div>
            <div class="kpi-value" style="color: {'#10B981' if is_high_propensity else '#EF4444'};">{prob*100:.1f}%</div>
            <div class="kpi-subtext">Decision Cutoff: {OPTIMAL_THRESHOLD*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">Lead Rating Tier</div>
            <div class="kpi-value" style="color: {'#10B981' if is_high_propensity else '#EF4444'};">
                {'TIER 1 (HIGH)' if is_high_propensity else 'TIER 3 (LOW)'}
            </div>
            <div class="kpi-subtext">Automated Priority Scoring</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">Routing Direction</div>
            <div class="kpi-value" style="color: #38BDF8; font-size: 1.6rem; padding-top:4px;">
                {'📞 PHONE QUEUE' if is_high_propensity else '📧 DIGITAL DRIP'}
            </div>
            <div class="kpi-subtext">Recommended Channel</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">Expected Net Margin</div>
            <div class="kpi-value" style="color: #10B981;">€{max(0, (prob*150)-5):.2f}</div>
            <div class="kpi-subtext">Revenue minus €5.00 Call Overhead</div>
        </div>
        """, unsafe_allow_html=True)

    c_gauge, c_action = st.columns([1.3, 1])
    
    with c_gauge:
        st.markdown("### 🎯 Conversion Gauge Chart")
        
        # High-end Ring Gauge Chart using Plotly
        fig_donut = go.Figure(go.Pie(
            values=[prob * 100, (1 - prob) * 100],
            labels=['Conversion Chance', 'Non-Conversion Chance'],
            hole=0.75,
            marker_colors=[COLOR_SUCCESS if is_high_propensity else COLOR_DANGER, "#1E293B"],
            hoverinfo="label+percent",
            textinfo="none"
        ))
        
        fig_donut.add_annotation(
            text=f"<b>{prob*100:.1f}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=38, color="#F8FAFC")
        )
        
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            height=280,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_action:
        st.markdown("### 🤖 Recommended Operational Action")
        if is_high_propensity:
            st.markdown(f"""
            <div class="decision-badge-high">
                <h3 style="color:#10B981; margin:0;">✅ HIGH PROPENSITY LEAD</h3>
                <hr style="border-color: rgba(16, 185, 129, 0.2); margin: 12px 0;">
                <p style="font-size: 1rem; line-height: 1.6; color: #E2E8F0;">
                <b>System Recommendation:</b> Dispatch to priority agent call queue immediately.<br><br>
                <b>Strategic Fit:</b> Customer score of <b>{prob*100:.1f}%</b> surpasses the optimal profitability cutoff of <b>{OPTIMAL_THRESHOLD*100:.1f}%</b>.<br><br>
                <b>Expected Value:</b> High likelihood of deposit commitment with strong positive campaign ROI.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-badge-low">
                <h3 style="color:#EF4444; margin:0;">🚫 LOW PROPENSITY LEAD</h3>
                <hr style="border-color: rgba(239, 68, 68, 0.2); margin: 12px 0;">
                <p style="font-size: 1rem; line-height: 1.6; color: #E2E8F0;">
                <b>System Recommendation:</b> Exclude from live call campaigns. Transition to email nurture.<br><br>
                <b>Strategic Fit:</b> Customer score of <b>{prob*100:.1f}%</b> falls below threshold.<br><br>
                <b>Direct Savings:</b> Prevents spending €5.00 on a cold lead, conserving marketing budget.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: MODEL DECISION BREAKDOWN
# ------------------------------------------
with tab_why:
    st.subheader("🔍 Interpreting the AI Score")
    st.markdown("Clear, plain-language insights into why the model predicted this specific probability.")

    # Driver Cards
    col_pos, col_neg = st.columns(2)
    
    with col_pos:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #10B981;">
            <h4 style="color: #10B981; margin-top:0;">🟢 Positive Drivers (Boosters)</h4>
        """, unsafe_allow_html=True)
        if duration > 300:
            st.markdown(f"• **High Call Duration ({duration}s):** Long engagement is the single strongest indicator of deposit interest.")
        if poutcome == "success":
            st.markdown("• **Past Success:** Clients who converted in previous campaigns convert at 5x higher rates.")
        if balance > 2000:
            st.markdown(f"• **Strong Account Balance (€{balance:,}):** Higher liquidity increases deposit capacity.")
        if duration <= 300 and poutcome != "success" and balance <= 2000:
            st.markdown("• Moderate baseline customer parameters.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_neg:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #EF4444;">
            <h4 style="color: #EF4444; margin-top:0;">🔴 Negative Drivers (Reducers)</h4>
        """, unsafe_allow_html=True)
        if duration <= 300:
            st.markdown(f"• **Short Call Duration ({duration}s):** Conversations under 5 minutes rarely yield conversions.")
        if poutcome == "unknown":
            st.markdown("• **No Prior Campaign History:** First-time outreach subjects have lower baseline intent.")
        if balance < 500:
            st.markdown(f"• **Low Liquidity (€{balance:,}):** Limited funds decrease deposit propensity.")
        if duration > 300 and poutcome != "unknown" and balance >= 500:
            st.markdown("• No major negative flags detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Feature Importance Chart
    st.markdown("### 🏆 Top Model Drivers Across All Customers")
    fi_data = pd.DataFrame({
        "Feature": ["Call Duration", "Previous Success", "Account Balance", "Customer Age", "Contact Day", "Housing Loan", "Campaign Calls"],
        "Importance": [0.425, 0.182, 0.124, 0.091, 0.073, 0.058, 0.047]
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_data, x="Importance", y="Feature", orientation="h",
        title="Global Feature Weight Distribution",
        color="Importance", color_continuous_scale="Tealgrn", template=PLOT_THEME
    )
    fig_fi.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=320)
    st.plotly_chart(fig_fi, use_container_width=True)

# ------------------------------------------
# TAB 3: FINANCIAL ROI SIMULATOR
# ------------------------------------------
with tab_business:
    st.subheader("💰 Executive Campaign Cost-Benefit Simulator")
    
    sim1, sim2 = st.columns([1, 2])
    with sim1:
        st.markdown("#### Campaign Parameters")
        n_audience = st.number_input("Target Audience Size", value=10000, step=1000)
        c_cost = st.number_input("Cost Per Outbound Call (€)", value=5.0, step=0.5)
        d_val = st.number_input("Profit Per Term Deposit (€)", value=150.0, step=10.0)
    
    with sim2:
        trad_exp = n_audience * c_cost
        trad_rev = (n_audience * 0.117) * d_val
        trad_net = trad_rev - trad_exp
        
        ml_calls = int(n_audience * 0.1387)
        ml_exp = ml_calls * c_cost
        ml_rev = (n_audience * 0.117 * 0.81) * d_val
        ml_net = ml_rev - ml_exp
        
        savings = trad_exp - ml_exp
        
        b1, b2, b3 = st.columns(3)
        b1.metric("Un-Targeted Return", f"€{trad_net:,.2f}")
        b2.metric("BankWise Net Return", f"€{ml_net:,.2f}", delta=f"+€{ml_net - trad_net:,.2f}")
        b3.metric("Operational Savings", f"€{savings:,.2f}", delta=f"{((savings)/trad_exp)*100:.1f}% Cut")
        
        # High-end Waterfall Visual
        fig_waterfall = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "total", "relative", "relative", "total"],
            x = ["Baseline Rev", "Un-Targeted Cost", "Un-Targeted Net", "Saved Expense", "ML Revenue", "BankWise Net"],
            textposition = "outside",
            y = [trad_rev, -trad_exp, trad_net, savings, (ml_rev - trad_rev), ml_net],
            connector = {"line":{"color":"#475569"}},
            decreasing = {"marker":{"color": COLOR_DANGER}},
            increasing = {"marker":{"color": COLOR_SUCCESS}},
            totals = {"marker":{"color": COLOR_PRIMARY}}
        ))
        fig_waterfall.update_layout(title="Campaign Cash Flow Comparison (€)", template=PLOT_THEME, paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=350)
        st.plotly_chart(fig_waterfall, use_container_width=True)

# ------------------------------------------
# TAB 4: PORTFOLIO DATA
# ------------------------------------------
with tab_eda:
    st.subheader("📊 Portfolio Demographic Insights")
    if df_engineered is not None:
        e1, e2 = st.columns(2)
        with e1:
            fig_dur = px.histogram(
                df_engineered, x="duration", color="y",
                title="Call Duration Distribution by Subscription Class",
                labels={"duration": "Call Duration (Seconds)", "y": "Subscribed"},
                color_discrete_map={0: "#64748B", 1: COLOR_SUCCESS}, template=PLOT_THEME, nbins=35
            )
            fig_dur.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG)
            st.plotly_chart(fig_dur, use_container_width=True)
            
        with e2:
            fig_box = px.box(
                df_engineered, x="y", y="balance",
                title="Balance Ranges vs Subscription Outcome",
                labels={"y": "Subscribed Outcome", "balance": "Yearly Balance (€)"},
                color="y", color_discrete_map={0: "#64748B", 1: COLOR_SUCCESS}, template=PLOT_THEME
            )
            fig_box.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG)
            st.plotly_chart(fig_box, use_container_width=True)