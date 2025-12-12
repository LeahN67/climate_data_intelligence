# climate_intelligence_platform_v2.py
"""
GreenScope - Agentic Climate Intelligence Platform (Modular Streamlit App)

Features:
- Core engine: ClimateDataGenerator, DynamicCausalDiscovery, CounterfactualForecaster, InterventionEngine
- Agentic GoalManager: goal-directed behavior for pilots
- Plugin architecture: load organization-specific analytics or custom modules
- Standard dashboard vs. custom module switch
- Lightweight self-update (model re-fit) demonstration

Run:
    streamlit run climate_intelligence_platform_v2.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
import importlib
import types
import warnings
import json
from typing import Dict, Any, List

warnings.filterwarnings("ignore")

# ---------------------------
# Page config & styling
# ---------------------------
st.set_page_config(page_title="GreenScope - Agentic Climate Intelligence", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8fafc; }
h1, h2, h3 { color: #0b4b8a; }
.block { background-color: white; padding: 14px; border-radius: 8px; box-shadow: 0 1px 8px rgba(0,0,0,0.04); }
</style>
""", unsafe_allow_html=True)

st.title("🌍 GreenScope — Agentic Climate Intelligence")
st.write("Prototype — Core engine + Plugin API (standard dashboard + org-specific modules)")

# ---------------------------
# 0. Plugin interface helpers
# ---------------------------

class PluginBase:
    """
    Base class for organization-specific plugins.
    A plugin should implement:
      - name: str
      - customize_ui(st)  # optional: add UI controls in sidebar
      - postprocess_results(results, data, context) -> dict  # optional: add custom outputs
    """
    name = "base_plugin"

    def customize_ui(self, st_module):
        pass

    def postprocess_results(self, results: Dict[str, Any], data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        # Default: no changes
        return results

def load_plugin(module_name: str) -> PluginBase:
    """
    Load a plugin by module name (must be importable).
    For early dev you can create a plugin file, e.g., plugins/kenya_plugin.py with a class KenyaPlugin(PluginBase)
    and set module_name = "plugins.kenya_plugin".
    """
    try:
        mod = importlib.import_module(module_name)
        # plugin expected to expose `Plugin` class
        if hasattr(mod, "Plugin"):
            plugin_cls = getattr(mod, "Plugin")
            plugin = plugin_cls()
            return plugin
    except Exception as e:
        st.warning(f"Could not load plugin '{module_name}': {e}")
    # fallback
    return PluginBase()

# ---------------------------
# 1. Perception: Data generator (same idea as before)
# ---------------------------

class ClimateDataGenerator:
    def __init__(self, n_years: int = 15, seed: int = 42):
        self.n_years = n_years
        self.n_months = n_years * 12
        np.random.seed(seed)

    def generate(self, scenario: str = "variable", region_type: str = "tropical") -> pd.DataFrame:
        months = np.arange(self.n_months)
        data = pd.DataFrame({"month": months})
        data["month_of_year"] = months % 12

        # climate drivers
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

        # temperature
        temp_season = p["temp_base"] + p["temp_amp"] * np.sin(2 * np.pi * data["month_of_year"] / 12)
        warming_trend = 0.01 * months / 12
        data["temperature"] = temp_season + warming_trend + np.random.normal(0, 1.5, self.n_months)

        # rainfall
        rain_driver_effect = 0.3 * data["enso"] + 0.25 * data["iod"]
        rain_variability = 2.0 if scenario == "variable" else 1.0
        data["rainfall"] = np.maximum(
            0,
            p["rain_base"]
            + 20 * rain_driver_effect
            + np.random.normal(0, 15 * rain_variability, self.n_months),
        )

        # climate state
        state = np.zeros(self.n_months)
        state[0] = 50
        for i in range(1, self.n_months):
            persistence = 0.7 * state[i - 1]
            input_flux = 0.4 * data["rainfall"].iloc[i]
            loss = 0.6 * (data["temperature"].iloc[i] - p["temp_base"])
            state[i] = np.clip(persistence + input_flux - loss + np.random.normal(0, 5), 0, 100)
        data["climate_state"] = state

        # impact index (stress)
        impact = np.zeros(self.n_months)
        impact[0] = 0.4
        for i in range(1, self.n_months):
            state_effect = 0.006 * state[i - 1]
            temp_stress = -0.012 * np.maximum(0, data["temperature"].iloc[i] - (p["temp_base"] + 8))
            impact[i] = np.clip(0.6 * impact[i - 1] + state_effect + temp_stress + np.random.normal(0, 0.025), 0.1, 0.9)
        data["impact_index"] = impact

        data["high_risk"] = (state < 30).astype(int)
        return data

# ---------------------------
# 2. Reasoning: Dynamic causal discovery
# ---------------------------

class DynamicCausalDiscovery:
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.changepoints: List[int] = []
        self.causal_graphs: List[Dict[str, Any]] = []

    def detect_changepoints(self, data: pd.DataFrame, variables: List[str], threshold: float = 0.25) -> List[int]:
        # Simpler, robust approach: rolling difference in means (fast)
        n = len(data)
        cps = []
        for i in range(self.window_size, n - self.window_size, 6):
            max_stat = 0
            for var in variables:
                before = data[var].iloc[i - self.window_size : i]
                after = data[var].iloc[i : i + self.window_size]
                stat = abs(before.mean() - after.mean()) / (np.std(before) + 1e-6)
                max_stat = max(max_stat, stat)
            if max_stat > threshold:
                cps.append(i)
        self.changepoints = cps
        return cps

    def granger_causality(self, x: np.ndarray, y: np.ndarray, max_lag: int = 3):
        # Keep a simple safe implementation for demo
        n = len(x)
        if n <= max_lag * 2:
            return 0.0, 1.0
        try:
            Xr = np.column_stack([np.roll(y, i) for i in range(1, max_lag + 1)])[max_lag:]
            yr = y[max_lag:]
            Xr = np.column_stack([np.ones(len(Xr)), Xr])
            betar = np.linalg.lstsq(Xr, yr, rcond=None)[0]
            rssr = np.sum((yr - Xr @ betar) ** 2)
            Xu = np.column_stack(
                [
                    Xr[:, 1:],
                    np.column_stack([np.roll(x, i) for i in range(1, max_lag + 1)])[max_lag:],
                ]
            )
            Xu = np.column_stack([np.ones(len(Xu)), Xu])
            betau = np.linalg.lstsq(Xu, yr, rcond=None)[0]
            rssu = np.sum((yr - Xu @ betau) ** 2)
            df1, df2 = max_lag, len(yr) - Xu.shape[1]
            if df2 <= 0 or rssu == 0:
                return 0.0, 1.0
            f_stat = ((rssr - rssu) / df1) / (rssu / df2)
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)
            return f_stat, p_val
        except Exception:
            return 0.0, 1.0

    def discover_graph(self, data: pd.DataFrame, variables: List[str], alpha: float = 0.05) -> Dict[str, Any]:
        n_vars = len(variables)
        adjacency = np.zeros((n_vars, n_vars))
        for i, var_y in enumerate(variables):
            for j, var_x in enumerate(variables):
                if i != j:
                    f_stat, p_val = self.granger_causality(data[var_x].values, data[var_y].values)
                    if p_val < alpha:
                        adjacency[j, i] = 1
        return {"variables": variables, "adjacency": adjacency}

    def run(self, data: pd.DataFrame, variables: List[str]):
        cps = self.detect_changepoints(data, variables)
        segments = [0] + cps + [len(data)]
        graphs = []
        for i in range(len(segments) - 1):
            start, end = segments[i], segments[i + 1]
            if end - start >= self.window_size:
                regime_data = data.iloc[start:end]
                graph = self.discover_graph(regime_data, variables)
                graphs.append({"regime_id": i, "start": start, "end": end, "graph": graph})
        self.causal_graphs = graphs
        return graphs

# ---------------------------
# 3. Simulation: Counterfactual forecaster
# ---------------------------

class CounterfactualForecaster:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_structural_equations(self, data: pd.DataFrame, causal_graph: Dict[str, Any]):
        variables = causal_graph["variables"]
        adjacency = causal_graph["adjacency"]
        if len(data) < 5:
            # not enough data to fit stable structural equations
            for var in variables:
                self.models[var] = {"type": "exogenous", "mean": 0.0, "std": 1.0}
            self.fitted = True
            return

        data_norm = pd.DataFrame(self.scaler.fit_transform(data[variables]), columns=variables)
        for i, var in enumerate(variables):
            parents = [variables[j] for j in range(len(variables)) if adjacency[j, i] == 1]
            if not parents:
                self.models[var] = {"type": "exogenous", "mean": data_norm[var].mean(), "std": data_norm[var].std() + 1e-6}
            else:
                X = data_norm[parents].values
                y = data_norm[var].values
                if len(y) > 1:
                    X_lagged = np.column_stack([X[1:], y[:-1]])
                    y_lagged = y[1:]
                else:
                    X_lagged, y_lagged = X, y
                X_with_intercept = np.column_stack([np.ones(len(X_lagged)), X_lagged])
                try:
                    beta = np.linalg.lstsq(X_with_intercept, y_lagged, rcond=None)[0]
                    residuals = y_lagged - X_with_intercept @ beta
                    residual_std = np.std(residuals) if len(residuals) > 0 else 0.1
                except Exception:
                    beta = np.zeros(X_with_intercept.shape[1])
                    residual_std = 0.5
                self.models[var] = {"type": "endogenous", "parents": parents, "coefficients": beta, "residual_std": residual_std}
        self.fitted = True

    def simulate_forward(self, initial_state: Dict[str, float], n_steps: int, interventions: Dict[str, float] = None):
        variables = list(self.models.keys())
        trajectories = {var: [initial_state.get(var, 0.0)] for var in variables}
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
                    # interventions are multiplicative adjustments
                    value = value * interventions[var]
                trajectories[var].append(value)
        return pd.DataFrame(trajectories)

    def forecast_risk(self, data: pd.DataFrame, causal_graph: Dict[str, Any], forecast_horizon: int = 6):
        # fit and simulate
        self.fit_structural_equations(data, causal_graph)
        variables = causal_graph["variables"]
        risk_var = "climate_state"
        last_obs = data[variables].iloc[-1].to_dict()
        # normalize using scaler; if not fitted defaults are handled in simulate
        try:
            last_norm = {var: (last_obs[var] - self.scaler.mean_[i]) / (self.scaler.scale_[i] + 1e-6) for i, var in enumerate(variables)}
        except Exception:
            last_norm = {var: last_obs[var] for var in variables}
        n_simulations = 100
        risk_counts = np.zeros(forecast_horizon)
        for _ in range(n_simulations):
            traj = self.simulate_forward(last_norm, forecast_horizon + 1)
            # denorm if scaler available
            try:
                traj_denorm = pd.DataFrame(self.scaler.inverse_transform(traj), columns=variables)
            except Exception:
                traj_denorm = traj
            if risk_var in traj_denorm.columns:
                risk_events = (traj_denorm[risk_var] < 30).astype(int).values[1:]
                risk_counts += risk_events
        return {"risk_probability": risk_counts / n_simulations}

# ---------------------------
# 4. Action: Intervention engine
# ---------------------------

class InterventionEngine:
    def __init__(self):
        self.interventions = {
            "enhanced_monitoring": {"effect": "climate_state", "desc": "Deploy targeted sensors & community monitoring"},
            "adaptive_management": {"effect": "impact_index", "desc": "Dynamic resource allocation protocols"},
            "early_action": {"effect": "impact_index", "desc": "Pre-emptive response triggers"},
        }
        self.costs = {"enhanced_monitoring": 120, "adaptive_management": 80, "early_action": 40}
        self.effects = {"enhanced_monitoring": 0.25, "adaptive_management": 0.20, "early_action": 0.15}

    def evaluate(self, forecaster: CounterfactualForecaster, data: pd.DataFrame, graph: Dict[str, Any], name: str, intensity: float = 0.5):
        baseline = forecaster.forecast_risk(data, graph)
        f_int = CounterfactualForecaster()
        f_int.fit_structural_equations(data, graph)
        vars_ = graph["variables"]
        last = data[vars_].iloc[-1].to_dict()
        try:
            last_norm = {v: (last[v] - f_int.scaler.mean_[i]) / (f_int.scaler.scale_[i] + 1e-6) for i, v in enumerate(vars_)}
        except Exception:
            last_norm = last
        effect_var = self.interventions[name]["effect"]
        mult = 1.0 + self.effects[name] * intensity
        interventions = {effect_var: mult}
        risk_counts = np.zeros(6)
        n_runs = 100
        for _ in range(n_runs):
            traj = f_int.simulate_forward(last_norm, 7, interventions)
            try:
                traj_denorm = pd.DataFrame(f_int.scaler.inverse_transform(traj), columns=vars_)
            except Exception:
                traj_denorm = traj
            events = (traj_denorm["climate_state"] < 30).astype(int).values[1:]
            risk_counts += events
        int_prob = risk_counts / n_runs
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

    def rank(self, forecaster: CounterfactualForecaster, data: pd.DataFrame, graph: Dict[str, Any], budget: float):
        results = []
        for name in self.interventions:
            for intensity in [0.3, 0.5, 0.7, 1.0]:
                res = self.evaluate(forecaster, data, graph, name, intensity)
                if res["cost"] <= budget:
                    results.append(res)
        return sorted(results, key=lambda x: x["effectiveness"], reverse=True)

# ---------------------------
# 5. Agentic GoalManager
# ---------------------------

class GoalManager:
    """
    Simple goal manager for agentic behavior.
    Goals are dicts with 'id', 'target' (e.g., reduce_risk_below=0.2), 'priority'
    """
    def __init__(self, goals: List[Dict[str, Any]] = None):
        self.goals = goals or []

    def add_goal(self, goal: Dict[str, Any]):
        self.goals.append(goal)
        # sort by priority (higher first)
        self.goals = sorted(self.goals, key=lambda g: g.get("priority", 1), reverse=True)

    def get_active_goal(self) -> Dict[str, Any]:
        return self.goals[0] if self.goals else {}

    def evaluate_progress(self, metric_name: str, value: float) -> Dict[str, Any]:
        # returns status compared to active goal; very simple
        goal = self.get_active_goal()
        if not goal:
            return {"status": "no_goal"}
        target = goal.get("target", {})
        if metric_name in target:
            target_val = target[metric_name]
            if value <= target_val:
                return {"status": "achieved", "goal": goal}
            else:
                return {"status": "pending", "distance": value - target_val, "goal": goal}
        return {"status": "metric_not_found"}

# ---------------------------
# 6. Core orchestrator + caching
# ---------------------------

@st.cache_resource
def run_core_engine(seed: int, years: int, region: str):
    data = ClimateDataGenerator(years, seed).generate("variable", region)
    variables = ["enso", "iod", "rainfall", "temperature", "climate_state", "impact_index"]
    discoverer = DynamicCausalDiscovery(window_size=24)
    graphs = discoverer.run(data, variables)
    recent = graphs[-1] if graphs else {"graph": {"variables": variables, "adjacency": np.zeros((6, 6))}, "start": 0, "end": len(data)}
    recent_data = data.iloc[recent["start"] : recent["end"]]
    forecaster = CounterfactualForecaster()
    forecast = forecaster.forecast_risk(recent_data, recent["graph"])
    return {"data": data, "discoverer": discoverer, "recent": recent, "recent_data": recent_data, "forecast": forecast}

# ---------------------------
# 7. UI + Orchestration
# ---------------------------

def main():
    st.sidebar.header("Pilot Config")
    # standard vs custom
    mode = st.sidebar.selectbox("Deployment Mode", ["Standard Dashboard", "Custom Organization Module"])
    region = st.sidebar.selectbox("Region Type", ["tropical", "arid", "mediterranean", "temperate"])
    years = st.sidebar.slider("Years of Data", 5, 20, 15)
    seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=42)
    st.sidebar.markdown("---")

    # Goal manager UI
    st.sidebar.header("Agentic Goals")
    goal_reduce = st.sidebar.slider("Target: reduce risk probability threshold (e.g., <= )", 0.05, 0.9, 0.3, 0.05)
    add_goal_btn = st.sidebar.button("Set Goal: Reduce risk probability")
    # instantiate goal manager
    gm = GoalManager()
    if add_goal_btn:
        gm.add_goal({"id": "reduce_risk", "target": {"risk_probability": goal_reduce}, "priority": 10})
        st.sidebar.success(f"Goal added: reduce risk probability <= {goal_reduce:.2f}")

    # plugin selection (module path)
    plugin_module = st.sidebar.text_input("Plugin module (optional)", value="", help="Enter python module path for custom module, e.g., plugins.kenya_plugin")
    plugin = load_plugin(plugin_module) if plugin_module else PluginBase()
    # allow plugin to add UI
    try:
        plugin.customize_ui(st.sidebar)
    except Exception as e:
        st.sidebar.warning(f"Plugin UI error: {e}")

    # run core engine
    core_results = run_core_engine(seed, years, region)
    data = core_results["data"]
    discoverer = core_results["discoverer"]
    recent = core_results["recent"]
    recent_data = core_results["recent_data"]
    forecast = core_results["forecast"]

    # standard dashboard main view
    st.header("Standard Insights")
    cols = st.columns([1, 2])
    with cols[0]:
        st.subheader("Quick Pilot Summary")
        st.write(f"Region: **{region.title()}** — Years: **{years}** — Seed: **{seed}**")
        st.write(f"Current regime: {recent.get('regime_id', 'n/a')} (rows {recent.get('start')}–{recent.get('end')})")
        st.write("Forecast (6 months):")
        st.metric("Avg risk next 6 months", f"{np.mean(forecast['risk_probability']):.1%}")
        # show active goal status
        goal_status = gm.evaluate_progress("risk_probability", float(np.mean(forecast["risk_probability"])))
        st.write("Goal status:", goal_status)

    with cols[1]:
        st.subheader("Top Recommendation (auto)")
        engine = InterventionEngine()
        rankings = engine.rank(CounterfactualForecaster(), recent_data, recent["graph"], budget=500)
        if rankings:
            top = rankings[0]
            st.markdown(f"**{top['name'].replace('_',' ').title()}** — {top['desc']}")
            st.write(f"Estimated cost: ${top['cost']:.0f} — Efficiency: {top['effectiveness']:.1f}")
        else:
            st.write("No actionable recommendations within the default budget.")

    # Visualization tabs (standard)
    st.subheader("Diagnostics")
    tab1, tab2, tab3 = st.tabs(["Climate Drivers", "Causal Graph", "Trends"])
    with tab1:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0, 0].plot(data["month"], data["enso"], label="ENSO", alpha=0.9)
        ax[0, 0].plot(data["month"], data["iod"], label="IOD", alpha=0.9)
        ax[0, 0].legend()
        ax[0, 0].set_title("Global Drivers")
        ax[0, 1].fill_between(data["month"], data["rainfall"], alpha=0.6)
        ax[0, 1].set_title("Precipitation")
        colors = ["#ef4444" if r else "#10b981" for r in data["high_risk"]]
        ax[1, 0].scatter(data["month"], data["climate_state"], c=colors, alpha=0.6, s=10)
        ax[1, 0].axhline(30, color="#dc2626", linestyle="--")
        ax[1, 0].set_title("Climate State & Risks")
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
            nx.draw(G, pos, with_labels=True, node_color="#c7e0ff", node_size=2000, font_size=10, arrowsize=14, edge_color="#1f4ed8")
            st.pyplot(fig)
        else:
            st.info("No significant causal relationships detected in the current regime.")

    with tab3:
        st.line_chart(data.set_index("month")[["rainfall", "temperature", "climate_state", "impact_index"]], height=300)

    # Forecast probabilities (standard)
    st.subheader("6-Month Risk Outlook")
    probs = forecast["risk_probability"]
    cols_p = st.columns(6)
    for i, (col, p) in enumerate(zip(cols_p, probs)):
        if p > 0.6:
            emoji = "🔴"
        elif p > 0.3:
            emoji = "🟠"
        else:
            emoji = "🟢"
        col.markdown(f"#### {emoji} Month +{i+1}\n**{p:.0%}**")

    # allow postprocessing by plugin (custom outputs)
    context = {"region": region, "years": years, "seed": seed}
    base_results = {"forecast": forecast, "recent_graph": recent["graph"], "recent_data": recent_data}
    results = plugin.postprocess_results(base_results, data, context) if isinstance(plugin, PluginBase) else base_results

    # Custom module UI (if selected)
    if mode == "Custom Organization Module":
        st.markdown("---")
        st.header("Custom Organization Module")
        st.info("You are running a custom organization plugin. Use the plugin to tailor metrics, KPIs, and exportable reports.")
        # show plugin results if any
        st.write("Plugin outputs:")
        st.json(results if isinstance(results, dict) else {"result": str(results)})

    # Standard exportable brief (downloadable)
    st.markdown("---")
    st.header("Export")
    brief = {
        "region": region,
        "years": years,
        "seed": seed,
        "avg_risk_next_6m": float(np.mean(forecast["risk_probability"])),
        "top_recommendation": rankings[0]["name"] if rankings else None,
    }
    st.download_button("Download Pilot Brief (JSON)", data=json.dumps(brief, indent=2), file_name="pilot_brief.json", mime="application/json")

    # Simple self-update demonstration: re-fit forecaster with latest data if requested
    st.sidebar.markdown("---")
    st.sidebar.header("Model Lifecycle")
    refit = st.sidebar.button("Refit forecaster with latest data (self-update)")
    if refit:
        st.sidebar.info("Refitting models... this emulates model update with new observations.")
        f_update = CounterfactualForecaster()
        f_update.fit_structural_equations(recent_data, recent["graph"])
        st.sidebar.success("Refit complete. Models updated for next forecasts.")

    # short feedback & next actions
    st.markdown("---")
    st.write("### Next steps (for pilot scale-up)")
    st.write(
        """
        1. Validate the attribution brief with local ministry experts.\n
        2. Scope a 6-12 month deployment (data-sharing, monitoring, and capacity building).\n
        3. If desired, replace the `PluginBase` with an organization-specific plugin module that adds KPI logic and export templates.
        """
    )

if __name__ == "__main__":
    main()
