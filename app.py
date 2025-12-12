# climate_intelligence_platform_public_private.py
"""
GreenScope — Agentic Climate Intelligence
Public demo analytics + Sign-in gated Custom Insights
Run: streamlit run climate_intelligence_platform_public_private.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import json

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
# Sidebar: controls + sign-in
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
    st.markdown("Sign in to unlock organisation-specific dashboards, downloads and API access.")
    if "signed_in" not in st.session_state:
        st.session_state["signed_in"] = False

    if st.button("Sign in for Custom Insights"):
        # Toggle sign-in (in real app, replace with real auth)
        st.session_state["signed_in"] = True

    if st.session_state["signed_in"]:
        st.success("Signed in: Custom insights unlocked")
        # small org settings (demo)
        org_name = st.text_input("Organisation name", value="My Ministry / Agency / NGO")
        org_scope = st.selectbox("Organisation scope", ["County", "National", "Regional", "Donor"])
    else:
        st.info("Custom insights are private to your organisation.")

st.markdown("")  # spacing

# -------------------------
# Core modules (unchanged core logic)
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
        data["enso"] = 1.5 * np.sin(enso_frequency * months) + np.random.normal(0, 0.5, self.n_months)
        iod_frequency = 2 * np.pi / 36
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


class DynamicCausalDiscovery:
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.changepoints = []
        self.causal_graphs = []

    def detect_changepoints(self, data, variables, threshold=2.0):
        n = len(data)
        changepoints = []
        for i in range(self.window_size, n - self.window_size, 6):
            max_stat = 0
            for var in variables:
                before = data[var].iloc[i - self.window_size : i]
                after = data[var].iloc[i : i + self.window_size]
                stat, _ = stats.ks_2samp(before, after)
                max_stat = max(max_stat, stat)
            if max_stat > threshold:
                changepoints.append(i)
        self.changepoints = changepoints
        return changepoints

    def granger_causality(self, x, y, max_lag=3):
        n = len(x)
        if n <= max_lag * 2:
            return 0, 1.0
        Xr = np.column_stack([np.roll(y, i) for i in range(1, max_lag + 1)])[max_lag:]
        yr = y[max_lag:]
        Xr = np.column_stack([np.ones(len(Xr)), Xr])
        br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        rss_r = np.sum((yr - Xr @ br) ** 2)
        Xu = np.column_stack(
            [
                Xr[:, 1:],
                np.column_stack([np.roll(x, i) for i in range(1, max_lag + 1)])[max_lag:],
            ]
        )
        Xu = np.column_stack([np.ones(len(Xu)), Xu])
        bu = np.linalg.lstsq(Xu, yr, rcond=None)[0]
        rss_u = np.sum((yr - Xu @ bu) ** 2)
        df1, df2 = max_lag, len(yr) - Xu.shape[1]
        if rss_u == 0 or df2 <= 0:
            return 0, 1.0
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        return f_stat, p_val

    def discover_graph(self, data, variables, alpha=0.05):
        n_vars = len(variables)
        adjacency = np.zeros((n_vars, n_vars))
        for i, var_y in enumerate(variables):
            for j, var_x in enumerate(variables):
                if i != j:
                    f_stat, p_val = self.granger_causality(data[var_x].values, data[var_y].values)
                    if p_val < alpha:
                        adjacency[j, i] = 1
        return {"variables": variables, "adjacency": adjacency}

    def run(self, data, variables):
        changepoints = self.detect_changepoints(data, variables)
        segments = [0] + changepoints + [len(data)]
        graphs = []
        for i in range(len(segments) - 1):
            start, end = segments[i], segments[i + 1]
            if end - start >= self.window_size:
                regime_data = data.iloc[start:end]
                graph = self.discover_graph(regime_data, variables)
                graphs.append({"regime_id": i, "start": start, "end": end, "graph": graph})
        self.causal_graphs = graphs
        return graphs


class CounterfactualForecaster:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def fit_structural_equations(self, data, causal_graph):
        variables = causal_graph["variables"]
        adjacency = causal_graph["adjacency"]
        data_norm = pd.DataFrame(self.scaler.fit_transform(data[variables]), columns=variables)
        for i, var in enumerate(variables):
            parents = [variables[j] for j in range(len(variables)) if adjacency[j, i] == 1]
            if not parents:
                self.models[var] = {"type": "exogenous", "mean": data_norm[var].mean(), "std": data_norm[var].std()}
            else:
                X = data_norm[parents].values
                y = data_norm[var].values
                if len(y) > 1:
                    X_lagged = np.column_stack([X[1:], y[:-1]])
                    y_lagged = y[1:]
                else:
                    X_lagged, y_lagged = X, y
                X_with_intercept = np.column_stack([np.ones(len(X_lagged)), X_lagged])
                beta = np.linalg.lstsq(X_with_intercept, y_lagged, rcond=None)[0]
                residuals = y_lagged - X_with_intercept @ beta
                residual_std = np.std(residuals) if len(residuals) > 0 else 0.1
                self.models[var] = {"type": "endogenous", "parents": parents, "coefficients": beta, "residual_std": residual_std}

    def simulate_forward(self, initial_state, n_steps, interventions=None):
        variables = list(self.models.keys())
        trajectories = {var: [initial_state[var]] for var in variables}
        for t in range(1, n_steps):
            for var in variables:
                model = self.models[var]
                if model["type"] == "exogenous":
                    value = np.random.normal(model["mean"], model["std"])
                else:
                    parent_values = [trajectories[p][-1] for p in model["parents"]]
                    lag_value = trajectories[var][-1]
                    X = np.concatenate([[1], parent_values, [lag_value]])
                    value = X @ model["coefficients"] + np.random.normal(0, model["residual_std"])
                if interventions and var in interventions:
                    value = value * interventions[var]
                trajectories[var].append(value)
        return pd.DataFrame(trajectories)

    def forecast_risk(self, data, causal_graph, forecast_horizon=6):
        self.fit_structural_equations(data, causal_graph)
        variables = causal_graph["variables"]
        risk_var = "climate_state"
        last_obs = data[variables].iloc[-1].to_dict()
        last_obs_norm = {var: (last_obs[var] - self.scaler.mean_[i]) / self.scaler.scale_[i] for i, var in enumerate(variables)}
        n_simulations = 100
        risk_counts = np.zeros(forecast_horizon)
        for _ in range(n_simulations):
            traj = self.simulate_forward(last_obs_norm, forecast_horizon + 1)
            traj_denorm = pd.DataFrame(self.scaler.inverse_transform(traj), columns=variables)
            if risk_var in traj_denorm.columns:
                risk_events = (traj_denorm[risk_var] < 30).astype(int).values[1:]
                risk_counts += risk_events
        return {"risk_probability": risk_counts / n_simulations}


# Intervention engine (as before)
class InterventionEngine:
    def __init__(self):
        self.interventions = {
            "enhanced_monitoring": {"effect": "climate_state", "desc": "Deploy targeted sensors & community monitoring"},
            "adaptive_management": {"effect": "impact_index", "desc": "Dynamic resource allocation protocols"},
            "early_action": {"effect": "impact_index", "desc": "Pre-emptive response triggers"},
        }
        self.costs = {"enhanced_monitoring": 120, "adaptive_management": 80, "early_action": 40}
        self.effects = {"enhanced_monitoring": 0.25, "adaptive_management": 0.20, "early_action": 0.15}

    def evaluate(self, forecaster, data, graph, name, intensity=0.5):
        baseline = forecaster.forecast_risk(data, graph)
        f_int = CounterfactualForecaster()
        f_int.fit_structural_equations(data, graph)
        vars_ = graph["variables"]
        last = data[vars_].iloc[-1].to_dict()
        last_norm = {v: (last[v] - f_int.scaler.mean_[i]) / f_int.scaler.scale_[i] for i, v in enumerate(vars_)}
        effect_var = self.interventions[name]["effect"]
        mult = 1.0 + self.effects[name] * intensity
        interventions = {effect_var: mult}
        risk_counts = np.zeros(6)
        for _ in range(100):
            traj = f_int.simulate_forward(last_norm, 7, interventions)
            traj_denorm = pd.DataFrame(f_int.scaler.inverse_transform(traj), columns=vars_)
            events = (traj_denorm["climate_state"] < 30).astype(int).values[1:]
            risk_counts += events
        int_prob = risk_counts / 100
        cost = self.costs[name] * intensity
        reduction = baseline["risk_probability"] - int_prob
        effectiveness = np.sum(reduction * 30) / (cost / 100) if cost > 0 else 0
        return {
            "name": name,
            "baseline": baseline["risk_probability"],
            "intervention": int_prob,
            "reduction": reduction,
            "cost": cost,
            "effectiveness": effectiveness,
            "desc": self.interventions[name]["desc"],
            "intensity": intensity,
        }

    def rank(self, forecaster, data, graph, budget):
        results = []
        for name in self.interventions:
            for intensity in [0.3, 0.5, 0.7, 1.0]:
                res = self.evaluate(forecaster, data, graph, name, intensity)
                if res["cost"] <= budget:
                    results.append(res)
        return sorted(results, key=lambda x: x["effectiveness"], reverse=True)


# -------------------------
# Run analysis (cached)
# -------------------------
@st.cache_resource
def run_analysis(seed, years, region):
    np.random.seed(seed)
    data = ClimateDataGenerator(years, seed).generate("variable", region)
    vars_ = ["enso", "iod", "rainfall", "temperature", "climate_state", "impact_index"]
    discoverer = DynamicCausalDiscovery(24)
    graphs = discoverer.run(data, vars_)
    recent = graphs[-1] if graphs else {"graph": {"variables": vars_, "adjacency": np.zeros((6, 6))}, "start": 0, "end": len(data)}
    recent_data = data.iloc[recent["start"] : recent["end"]]
    forecaster = CounterfactualForecaster()
    forecast = forecaster.forecast_risk(recent_data, recent["graph"])
    return data, discoverer, recent, recent_data, forecast


# -------------------------
# Execute pipeline and render public analytics
# -------------------------
data, discoverer, recent, recent_data, forecast = run_analysis(seed, years, region)

# Public analytics: risk outlook + intervention impact + diagnostics (available without sign-in)
st.subheader("🌦️ 6-Month Climate Risk Outlook (Public)")

probs = forecast["risk_probability"]
cols = st.columns(6)
for i, (col, p) in enumerate(zip(cols, probs)):
    if p > 0.6:
        emoji = "🔴"
        color = "#EF4444"
    elif p > 0.3:
        emoji = "🟠"
        color = "#F59E0B"
    else:
        emoji = "🟢"
        color = "#10B981"
    col.markdown(f"<div style='text-align:center; color:{color}; font-weight:700;'>{emoji}<br>+{i+1}<br>{p:.0%}</div>", unsafe_allow_html=True)

engine = InterventionEngine()
eval_res = engine.evaluate(CounterfactualForecaster(), recent_data, recent["graph"], intervention, intensity)

st.subheader("🛠️ Intervention Snapshot (Public)")
c1, c2, c3 = st.columns(3)
c1.metric("Cost (est.)", f"${eval_res['cost']:.0f}")
c2.metric("Avg Risk Reduction", f"{np.mean(eval_res['reduction']):.0%}")
c3.metric("Impact Efficiency", f"{eval_res['effectiveness']:.1f} risk-days/$100")
st.caption(f"*{eval_res['desc']}*")

rankings = engine.rank(CounterfactualForecaster(), recent_data, recent["graph"], budget)
if rankings:
    st.subheader("Top Recommendations (sample, Public)")
    for i, rec in enumerate(rankings[:2]):  # show only top 2 publicly
        with st.expander(f"{i+1}. {rec['name'].replace('_',' ').title()} @ {rec['intensity']:.1f}x"):
            st.write(f"**Cost**: ${rec['cost']:.0f}")
            st.write(f"**Efficiency**: {rec['effectiveness']:.1f} risk-days per $100")
            st.write(f"**Risk Reduction**: {np.mean(rec['reduction']):.1%}")
            st.write(f"*{rec['desc']}*")

# Diagnostics tabs (public)
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
    graph = recent["graph"]
    G = nx.DiGraph()
    vars_list = graph["variables"]
    adj = graph["adjacency"]
    for i, src in enumerate(vars_list):
        for j, tgt in enumerate(vars_list):
            if adj[i, j] == 1:
                G.add_edge(src, tgt)
    if G.number_of_edges() > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="#DBEAFE", node_size=2000, font_size=10, arrowsize=14, edge_color="#1f4ed8")
        st.pyplot(fig)
    else:
        st.info("No significant causal relationships detected in the current regime.")

with tab3:
    st.line_chart(data.set_index("month")[["rainfall", "temperature", "climate_state", "impact_index"]], height=300)

# -------------------------
# SIGNED-IN / CUSTOM INSIGHTS SECTION (Gated)
# -------------------------
if st.session_state.get("signed_in", False):
    st.markdown("---")
    st.subheader("🔐 Custom Insights — Organisation View")

    # Organisation summary card
    org_display = locals().get("org_name", "Your Organisation") if st.session_state.get("signed_in") else "Your Organisation"
    st.markdown(f"<div class='card'><strong>Organisation:</strong> {org_display} &nbsp;&nbsp; <strong>Scope:</strong> {locals().get('org_scope','N/A')}</div>", unsafe_allow_html=True)

    # Custom KPIs (examples)
    st.markdown("### Organisation KPIs (customised)")
    k1, k2, k3 = st.columns(3)
    # simple derived KPIs for demo
    drought_prob = float(np.mean(forecast["risk_probability"]))
    k1.metric("6m Avg Drought Risk", f"{drought_prob:.1%}")
    k2.metric("Estimated Attributable Impact", f"{np.mean(data['impact_index']):.2f}")
    k3.metric("Stations Needed (est.)", f"{max(1, int(10 * (1 - np.mean(data['high_risk']))))}")

    st.markdown("### Custom Attribution Brief")
    st.markdown("Download a draft attribution brief (pre-filled) — edit and share with donors or internal teams.")
    brief = {
        "org": org_display,
        "region": region,
        "years": years,
        "avg_risk_next_6m": float(np.mean(forecast["risk_probability"])),
        "top_public_recommendation": (rankings[0]["name"] if rankings else None)
    }
    st.download_button("Download Attribution Brief (JSON)", data=json.dumps(brief, indent=2), file_name="attribution_brief.json", mime="application/json")

    st.markdown("### Organisation Tools")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**API Access**")
        st.code("API_KEY: <demo-key-XXXX-YYYY>", language="text")
        st.markdown("Use this key to access organisation endpoints (demo).")
    with c2:
        st.markdown("**Export**")
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download underlying data (CSV)", data=csv, file_name="climate_data.csv", mime="text/csv")

    st.markdown("### Custom Scenario Builder")
    st.markdown("Run a custom counterfactual for your programme:")
    cf_rain_adj = st.slider("Rainfall multiplier (scenario)", 0.5, 1.5, 1.0, 0.05)
    # simple simulation applying multiplier to rainfall in a copy
    custom_data = recent_data.copy()
    custom_data["rainfall_mod"] = custom_data["rainfall"] * cf_rain_adj
    st.markdown("Preview of modified rainfall (first 10 rows):")
    st.dataframe(custom_data[["rainfall", "rainfall_mod"]].head(10))

    # Custom note
    st.info("Custom insights are for authorised partners. Replace demo auth with your identity provider in production.")

else:
    # not signed in: show gentle CTA and short list of custom features
    st.markdown("---")
    st.info(
        """
        Want organisation-specific KPIs, attribution briefs, API access and exportable reports?
        Click **Sign in for Custom Insights** in the sidebar to unlock secure organisation dashboards.
        """
    )

st.markdown("---")
st.caption("GreenScope — prototype agentic climate intelligence. For pilot use only.")
