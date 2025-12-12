# -------------------------
# GreenScope — Agentic Climate Intelligence
# Public + Sign-in gated Custom Insights + PDF with charts
# -------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from scipy import stats
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(
    page_title="GreenScope Climate Intelligence",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #F7F9FB; font-family: 'Inter', sans-serif; }
    h1 { color: #1E3A8A; font-weight:700; }
    h2 { color: #059669; font-weight:700; }
    .card { background-color: #ffffff; padding:12px; border-radius:10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .muted { color:#6B7280; }
    .btn-primary { background-color:#059669; color:white; border-radius:8px; padding:8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Initialize session state
# -------------------------
if "signed_in" not in st.session_state:
    st.session_state["signed_in"] = False
if "org_name" not in st.session_state:
    st.session_state["org_name"] = None
if "org_scope" not in st.session_state:
    st.session_state["org_scope"] = None

# -------------------------
# Top navigation bar
# -------------------------
nav_col1, nav_col2, nav_col3 = st.columns([6, 1, 1])
with nav_col3:
    st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
    if not st.session_state["signed_in"]:
        if st.button("🔐 Sign in", key="top_sign_in"):
            st.session_state["signed_in"] = True
            st.session_state["org_name"] = "My Organisation"
    else:
        if st.button("🚪 Sign out"):
            st.session_state["signed_in"] = False
            st.session_state["org_name"] = None
            st.session_state["org_scope"] = None
    if st.session_state["signed_in"]:
        st.markdown(
            f"<div style='font-size:14px; color:#374151;'>✅ Signed in: <strong>{st.session_state['org_name']}</strong></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<h2>🌍 GreenScope — Agentic Climate Intelligence</h2>", unsafe_allow_html=True)
st.markdown(
    """<p class="muted">
    <strong>When data is sparse but stakes are high</strong>, our equity-centered platform uses causal AI to reveal hidden climate risks
    and simulate actionable futures—so decisions today can build resilience tomorrow.
    </p>""",
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Pilot Controls")
    region = st.selectbox("Region Type", ["tropical", "arid", "mediterranean", "temperate"])
    years = st.slider("Years of Data", 5, 20, 15)
    seed = st.slider("Random Seed", 1, 100, 42)

    st.markdown("---")
    st.header("Intervention Planner")
    budget = st.slider("Max Budget ($)", 100, 2000, 500)
    intervention = st.selectbox("Intervention", ["enhanced_monitoring", "adaptive_management", "early_action"])
    intensity = st.slider("Intensity", 0.1, 1.0, 0.5, 0.1)

    st.markdown("---")
    st.header("Custom Insights")
    if not st.session_state["signed_in"]:
        st.markdown("Sign in to unlock organisation-specific dashboards, downloads and API access.")
        if st.button("Sign in for Custom Insights"):
            st.session_state["signed_in"] = True
    else:
        st.success("Signed in: Custom insights unlocked")
        st.session_state["org_name"] = st.text_input("Organisation name", value=st.session_state.get("org_name","My Ministry / Agency / NGO"))
        st.session_state["org_scope"] = st.selectbox("Organisation scope", ["County", "National", "Regional", "Donor"])

# -------------------------
# Core modules
# -------------------------
class ClimateDataGenerator:
    def __init__(self, n_years=15, seed=42):
        self.n_years = n_years
        self.n_months = n_years * 12
        np.random.seed(seed)

    def generate(self, scenario="variable", region_type="tropical"):
        months = np.arange(self.n_months)
        data = pd.DataFrame({"month": months})
        data["month_of_year"] = months % 12
        enso_frequency = 2 * np.pi / 48
        iod_frequency = 2 * np.pi / 36
        data["enso"] = 1.5 * np.sin(enso_frequency * months) + np.random.normal(0, 0.5, self.n_months)
        data["iod"] = 1.2 * np.sin(iod_frequency * months + np.pi / 4) + np.random.normal(0, 0.4, self.n_months)

        params = {
            "tropical": {"temp_base": 26, "temp_amp": 3, "rain_base": 100},
            "arid": {"temp_base": 28, "temp_amp": 6, "rain_base": 30},
            "mediterranean": {"temp_base": 18, "temp_amp": 8, "rain_base": 60},
            "temperate": {"temp_base": 12, "temp_amp": 10, "rain_base": 80},
        }
        p = params.get(region_type, params["tropical"])

        temp_season = p["temp_base"] + p["temp_amp"] * np.sin(2 * np.pi * data["month_of_year"] / 12)
        warming_trend = 0.01 * months / 12
        data["temperature"] = temp_season + warming_trend + np.random.normal(0, 1.5, self.n_months)

        rain_driver_effect = 0.3 * data["enso"] + 0.25 * data["iod"]
        rain_variability = 2.0 if scenario == "variable" else 1.0
        data["rainfall"] = np.maximum(0, p["rain_base"] + 20 * rain_driver_effect + np.random.normal(0, 15 * rain_variability, self.n_months))

        state = np.zeros(self.n_months)
        state[0] = 50
        for i in range(1, self.n_months):
            persistence = 0.7 * state[i - 1]
            input_flux = 0.4 * data["rainfall"].iloc[i]
            loss = 0.6 * (data["temperature"].iloc[i] - p["temp_base"])
            state[i] = np.clip(persistence + input_flux - loss + np.random.normal(0, 5), 0, 100)
        data["climate_state"] = state

        impact = np.zeros(self.n_months)
        impact[0] = 0.4
        for i in range(1, self.n_months):
            state_effect = 0.006 * state[i - 1]
            temp_stress = -0.012 * np.maximum(0, data["temperature"].iloc[i] - (p["temp_base"] + 8))
            impact[i] = np.clip(0.6 * impact[i - 1] + state_effect + temp_stress + np.random.normal(0, 0.025), 0.1, 0.9)
        data["impact_index"] = impact

        data["high_risk"] = (state < 30).astype(int)
        return data

# -------------------------
# Run analysis
# -------------------------
@st.cache_resource
def run_analysis(seed, years, region):
    data = ClimateDataGenerator(years, seed).generate("variable", region)
    recent_data = data
    forecast = {"risk_probability": np.random.rand(6).tolist()}
    ranked_interventions = [
        {"name": "enhanced_monitoring", "desc": "Deploy targeted sensors & community monitoring",
         "cost": 120, "effectiveness": 0.8, "intensity": 0.5, "reduction": np.random.rand(6).tolist()},
        {"name": "adaptive_management", "desc": "Dynamic resource allocation protocols",
         "cost": 80, "effectiveness": 0.7, "intensity": 0.5, "reduction": np.random.rand(6).tolist()},
    ]
    return data, recent_data, forecast, ranked_interventions

data, recent_data, forecast, ranked = run_analysis(seed, years, region)

# -------------------------
# Public Tabs
# -------------------------
st.subheader("🔍 System Diagnostics (Public)")
tab1, tab2, tab3 = st.tabs(["Climate Drivers", "Causal Insights", "Trends"])

with tab1:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(data["month"], data["enso"], label="ENSO")
    ax[0, 0].plot(data["month"], data["iod"], label="IOD")
    ax[0, 0].legend()
    ax[0, 0].set_title("Global Climate Drivers")
    ax[0, 1].fill_between(data["month"], data["rainfall"], alpha=0.6)
    ax[0, 1].set_title("Precipitation")
    colors = ["#EF4444" if r else "#10B981" for r in data["high_risk"]]
    ax[1, 0].scatter(data["month"], data["climate_state"], c=colors, alpha=0.6, s=10)
    ax[1, 0].axhline(30, color="#DC2626", linestyle="--")
    ax[1, 0].set_title("Climate State & Risk Events")
    ax[1, 1].plot(data["month"], data["impact_index"])
    ax[1, 1].set_title("Impact Index")
    for a in ax.flat:
        a.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    # Placeholder causal graph
    G = nx.DiGraph()
    G.add_edges_from([("enso", "rainfall"), ("temperature", "climate_state")])
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="#DBEAFE", node_size=2000, font_size=10, arrowsize=14, edge_color="#1f4ed8")
    st.pyplot(fig)

with tab3:
    st.line_chart(data.set_index("month")[["rainfall", "temperature", "climate_state", "impact_index"]], height=300)

# -------------------------
# Signed-in Custom Insights + PDF
# -------------------------
if st.session_state["signed_in"]:
    st.markdown("---")
    st.subheader("🔐 Custom Insights — Organisation View")

    st.markdown(f"<div class='card'><strong>Organisation:</strong> {st.session_state['org_name']} &nbsp;&nbsp; <strong>Scope:</strong> {st.session_state['org_scope']}</div>", unsafe_allow_html=True)

    st.markdown("### Top Interventions (Custom)")
    for res in ranked:
        st.markdown(f"""
        **{res['name']}** — {res['desc']}
        - Cost: ${res['cost']:.0f}
        - Effectiveness: {res['effectiveness']:.2f}
        """)

    # PDF generation
    def generate_pdf(org_name, region, data, interventions):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"GreenScope Custom Report — {org_name}", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.multi_cell(0, 8, f"Region: {region}")
        pdf.ln(5)
        pdf.multi_cell(0, 8, "Top Interventions:")
        for res in interventions:
            pdf.multi_cell(0, 8, f"- {res['name']}: {res['desc']}, Cost: ${res['cost']}, Effectiveness: {res['effectiveness']}")
        # Charts
        buf = BytesIO()
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs[0,0].plot(data["month"], data["temperature"], color='red'); axs[0,0].set_title("Temperature")
        axs[0,1].plot(data["month"], data["rainfall"], color='blue'); axs[0,1].set_title("Rainfall")
        axs[1,0].plot(data["month"], data["climate_state"], color='green'); axs[1,0].set_title("Climate State")
        axs[1,1].plot(data["month"], data["impact_index"], color='orange'); axs[1,1].set_title("Impact Index")
        plt.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        pdf.image(buf, x=10, y=120, w=180)
        out = BytesIO()
        pdf.output(out)
        out.seek(0)
        return out

    pdf_report = generate_pdf(st.session_state["org_name"], region, data, ranked)
    st.download_button(
        label="📥 Download Custom Report with Charts (PDF)",
        data=pdf_report,
        file_name=f"greenscope_report_{st.session_state['org_name'].replace(' ','_')}.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.caption("GreenScope — prototype agentic climate intelligence. For pilot use only.")
