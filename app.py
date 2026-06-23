"""
Wind Turbine Health Monitoring Dashboard
==========================================
Bachelor Thesis: Contrastive Explanations for Wind Turbine Health
Monitoring Using SCADA Data -- Abdelrahman Fikri, GUC.

Required files/folders (place next to this script):

saved_models/
    lightgbm.pkl, xgboost.pkl, extra_trees.pkl, random_forest.pkl
    scaler.pkl, feature_cols.pkl
explainer_lgb.pkl, explainer_xgb.pkl, explainer_et.pkl, explainer_rf.pkl
shap_values_lgb.npy, shap_values_xgb.npy, shap_values_et.npy, shap_values_rf.npy
pfi_results.pkl
ale_lightgbm_top1.png        (kept static -- ALE only computed for 1 model/feature)
lime_{model}_{Service|Downtime|Other}.png   (kept static -- LIME needs X_train to recompute live)
dashboard_data/
    readable_feature_names.pkl, sensor_lookup.pkl, class_names.pkl
    X_test_scaled.csv, y_test.csv, master_df_test.csv
    X_shap_sample.csv, anomaly_instances.pkl

Global SHAP, Local SHAP, and Contrastive SHAP are computed LIVE in this
app from the loaded explainers + saved SHAP value arrays + anomaly
instances -- not read from static PNGs -- so they are fully interactive
(model switch, instance switch, hover tooltips) per the project's
request to favour interactivity wherever the underlying data supports it.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Turbine Health Monitor",
    page_icon="🌬️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

MODEL_FILES = {
    "LightGBM": "lightgbm.pkl",
    "XGBoost": "xgboost.pkl",
    "Extra Trees": "extra_trees.pkl",
    "Random Forest": "random_forest.pkl",
}
MODEL_SLUG = {
    "LightGBM": "lightgbm", "XGBoost": "xgboost",
    "Extra Trees": "extra_trees", "Random Forest": "random_forest",
}
MODEL_SLUG_SHORT = {
    "LightGBM": "lgb", "XGBoost": "xgb",
    "Extra Trees": "et", "Random Forest": "rf",
}
EXPLAINER_FILES = {
    "LightGBM": "explainer_lgb.pkl", "XGBoost": "explainer_xgb.pkl",
    "Extra Trees": "explainer_et.pkl", "Random Forest": "explainer_rf.pkl",
}
SHAP_VALUE_FILES = {
    "LightGBM": "shap_values_lgb.npy", "XGBoost": "shap_values_xgb.npy",
    "Extra Trees": "shap_values_et.npy", "Random Forest": "shap_values_rf.npy",
}

CLASS_NAMES_LIST = ["Normal", "Service", "Downtime", "Other"]
CLASS_COLORS = {
    "Normal": "#2ECC71", "Service": "#F39C12",
    "Downtime": "#E74C3C", "Other": "#9B59B6",
}
ANOMALY_INSTANCE_NAMES = ["Service", "Downtime", "Other"]  # exact casing used in saved filenames

ROLE_DESCRIPTIONS = {
    "Control Room Operator": "Real-time triage: which turbine is abnormal "
        "right now, and what is the top sensor driving the alert.",
    "Maintenance Engineer": "Diagnosis: detailed, plain-language explanations "
        "for a specific fault, to decide what to inspect.",
    "Operations Manager": "Aggregate trends: which sensors and turbines are "
        "most consistently fault-prone across the whole fleet.",
}
ROLE_TABS = {
    "Control Room Operator": ["Overview", "Time Series", "Prediction"],
    "Maintenance Engineer": ["Prediction", "Explanation"],
    "Operations Manager": ["Fleet Analytics"],
}


# ─────────────────────────────────────────────────────────────────────
# LOADING  (cached so files are read once per session)
# ─────────────────────────────────────────────────────────────────────
def safe_load_pickle(path):
    """Load a .pkl file whether it was written with joblib or plain
    pickle -- tries joblib first, falls back to pickle."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models, missing = {}, []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(SAVED_MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = safe_load_pickle(path)
        else:
            missing.append(path)
    return models, missing


@st.cache_resource(show_spinner="Loading SHAP explainers...")
def load_explainers():
    explainers, missing = {}, []
    for name, fname in EXPLAINER_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            explainers[name] = safe_load_pickle(path)
        else:
            missing.append(path)
    return explainers, missing


@st.cache_resource(show_spinner="Loading pre-computed SHAP value arrays...")
def load_shap_value_arrays():
    arrays, missing = {}, []
    for name, fname in SHAP_VALUE_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            arrays[name] = np.load(path, allow_pickle=True)
        else:
            missing.append(path)
    return arrays, missing


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata():
    meta, missing = {}, []
    paths = {
        "scaler": os.path.join(SAVED_MODELS_DIR, "scaler.pkl"),
        "feature_cols": os.path.join(SAVED_MODELS_DIR, "feature_cols.pkl"),
        "readable_feature_names": os.path.join(DATA_DIR, "readable_feature_names.pkl"),
        "sensor_lookup": os.path.join(DATA_DIR, "sensor_lookup.pkl"),
        "class_names": os.path.join(DATA_DIR, "class_names.pkl"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            meta[key] = safe_load_pickle(path)
        else:
            missing.append(path)
            meta[key] = None
    return meta, missing


@st.cache_resource(show_spinner="Loading PFI results...")
def load_pfi_results():
    path = os.path.join(BASE_DIR, "pfi_results.pkl")
    if os.path.exists(path):
        return safe_load_pickle(path), []
    return None, [path]


@st.cache_data(show_spinner="Loading test set...")
def load_test_data():
    data, missing = {}, []
    paths = {
        "X_test_scaled": os.path.join(DATA_DIR, "X_test_scaled.csv"),
        "y_test": os.path.join(DATA_DIR, "y_test.csv"),
        "master_df_test": os.path.join(DATA_DIR, "master_df_test.csv"),
        "X_shap_sample": os.path.join(DATA_DIR, "X_shap_sample.csv"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            missing.append(path)
            data[key] = None
    return data, missing


@st.cache_resource(show_spinner="Loading anomaly instances...")
def load_anomaly_instances():
    path = os.path.join(DATA_DIR, "anomaly_instances.pkl")
    if os.path.exists(path):
        return safe_load_pickle(path), []
    return None, [path]


def img_or_warning(path):
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Expected file not found: `{os.path.basename(path)}`")


# ─────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────
models, missing_models = load_models()
explainers, missing_explainers = load_explainers()
shap_value_arrays, missing_shap_arrays = load_shap_value_arrays()
meta, missing_meta = load_metadata()
pfi_results, missing_pfi = load_pfi_results()
test_data, missing_test = load_test_data()
anomaly_instances, missing_anomaly = load_anomaly_instances()

all_missing = (missing_models + missing_explainers + missing_shap_arrays +
               missing_meta + missing_pfi + missing_test + missing_anomaly)

feature_cols = meta.get("feature_cols") or []
scaler = meta.get("scaler")
readable_names = meta.get("readable_feature_names")
sensor_lookup = meta.get("sensor_lookup")
class_names = meta.get("class_names") or {0: "Normal", 1: "Service", 2: "Downtime", 3: "Other"}

X_test_scaled = test_data.get("X_test_scaled")
y_test = test_data.get("y_test")
master_df_test = test_data.get("master_df_test")

# A feature -> readable-name map restricted to ONLY the 861 features the
# models were actually trained on (feature_cols). This is what fixes
# the "sensor_16 / sensor_100-105 shown unlabeled" issue: those sensors
# are angle sensors that were already dropped from feature_cols during
# preprocessing (notebook Cell 12) -- they were never part of the model's
# input space, so they should never appear as selectable sensors here.
if readable_names and feature_cols and len(readable_names) == len(feature_cols):
    FEATURE_NAME_MAP = dict(zip(feature_cols, readable_names))
else:
    FEATURE_NAME_MAP = {c: c for c in feature_cols}

if master_df_test is not None and "asset_id" in master_df_test.columns:
    ALL_TURBINES = sorted(master_df_test["asset_id"].unique().tolist())
else:
    ALL_TURBINES = []


# ─────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────
def get_turbine_rows(asset_id):
    """Returns this turbine's rows from the test set, sorted into real
    chronological order using time_stamp (when available), so that
    'recording 1' means the same thing everywhere in the app."""
    if master_df_test is None or X_test_scaled is None:
        return None, None
    mask = master_df_test["asset_id"] == asset_id
    t_meta = master_df_test[mask]
    t_scaled = X_test_scaled[mask]
    if "time_stamp" in t_meta.columns:
        t_meta = t_meta.copy()
        t_meta["time_stamp"] = pd.to_datetime(t_meta["time_stamp"])
        order = t_meta["time_stamp"].argsort()
        t_meta = t_meta.iloc[order]
        t_scaled = t_scaled.iloc[order]
    return t_meta, t_scaled


def turbine_picker(key_suffix):
    if not ALL_TURBINES:
        st.warning("No turbines found in the test set (`master_df_test.csv` missing or empty).")
        return None
    return st.selectbox(
        "Select turbine", ALL_TURBINES,
        format_func=lambda x: f"Turbine {x}",
        key=f"turbine_picker_{key_suffix}"
    )


def compute_global_shap(model_name):
    """Mean |SHAP| per feature, averaged across all classes and all
    sampled instances. Returns (feature_names, importances) sorted
    descending, or (None, None) if the SHAP array isn't loaded."""
    arr = shap_value_arrays.get(model_name)
    if arr is None:
        return None, None
    mean_abs = np.mean(np.abs(arr), axis=(0, 2))  # (n_features,)
    names = readable_names if readable_names else feature_cols
    order = np.argsort(mean_abs)[::-1]
    return [names[i] for i in order], mean_abs[order]


def compute_instance_shap(model_name, instance_row):
    """Run the loaded TreeExplainer on a single instance. Returns
    (shap_array, pred_class, base_values) or (None, None, None) if the
    model/explainer for this name isn't loaded."""
    model = models.get(model_name)
    explainer = explainers.get(model_name)
    if model is None or explainer is None:
        return None, None, None
    pred_class = int(model.predict(instance_row)[0])
    shap_vals = explainer.shap_values(instance_row)  # shape (1, n_features, n_classes)
    return shap_vals, pred_class, explainer.expected_value


@st.cache_data(show_spinner="Computing how this sensor affects the prediction...")
def compute_ale_live(model_name, feature_col, pred_class, n_bins=12):
    """
    Compute a live Accumulated Local Effects (ALE) curve for one feature,
    for the probability of one specific class, using the fleet-wide test
    set as the reference population.

    For each bin of the feature's observed values, this looks at how the
    model's predicted probability for `pred_class` would change if every
    instance currently in that bin had its feature value moved from the
    bin's lower edge to its upper edge -- holding every other feature at
    its real, observed value (this "real joint distribution" property is
    what distinguishes ALE from a naive partial-dependence plot). These
    local changes are accumulated (cumulative sum) across bins and then
    centred so the curve averages to zero, matching the standard ALE
    definition.

    Returns (bin_centers_scaled, ale_values) or (None, None) if the
    model/data needed isn't available.
    """
    model = models.get(model_name)
    if model is None or X_test_scaled is None or feature_col not in X_test_scaled.columns:
        return None, None

    X = X_test_scaled[feature_cols].copy()
    x_vals = X[feature_col].values

    bin_edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, n_bins + 1)))
    if len(bin_edges) < 3:
        return None, None  # not enough distinct values to bin meaningfully
    n_actual_bins = len(bin_edges) - 1

    bin_idx = np.clip(np.digitize(x_vals, bin_edges[1:-1], right=True), 0, n_actual_bins - 1)

    local_effects = np.zeros(n_actual_bins)
    bin_counts = np.zeros(n_actual_bins)

    for k in range(n_actual_bins):
        mask = bin_idx == k
        n_in_bin = mask.sum()
        bin_counts[k] = n_in_bin
        if n_in_bin == 0:
            continue
        X_lower = X[mask].copy()
        X_upper = X[mask].copy()
        X_lower[feature_col] = bin_edges[k]
        X_upper[feature_col] = bin_edges[k + 1]
        pred_lower = model.predict_proba(X_lower)[:, pred_class]
        pred_upper = model.predict_proba(X_upper)[:, pred_class]
        local_effects[k] = np.mean(pred_upper - pred_lower)

    ale_uncentered = np.cumsum(local_effects)
    weights = bin_counts / bin_counts.sum()
    ale_centered = ale_uncentered - np.sum(ale_uncentered * weights)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, ale_centered


def to_real_units(scaled_values, feature_col):
    """Inverse-transform StandardScaler output back to real sensor units
    for one feature, given the fitted scaler and the feature's position
    in feature_cols. Returns the input unchanged if scaler/index unavailable."""
    if scaler is None or feature_col not in feature_cols:
        return scaled_values
    idx = feature_cols.index(feature_col)
    try:
        return np.asarray(scaled_values) * scaler.scale_[idx] + scaler.mean_[idx]
    except Exception:
        return scaled_values


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🌬️ Wind Turbine Monitor")
st.sidebar.markdown("---")

user_role = st.sidebar.selectbox(
    "Select Your Role",
    ["Control Room Operator", "Maintenance Engineer", "Operations Manager"]
)
visible_tabs = ROLE_TABS[user_role]

st.sidebar.caption(ROLE_DESCRIPTIONS[user_role])
st.sidebar.markdown("---")

# The model selector is only meaningful on tabs that actually use a
# specific model (Overview, Prediction, Explanation). Operations Manager
# only sees Fleet Analytics, which compares ALL models at once, so the
# model selector is hidden for that role to avoid implying it changes
# anything there.
if user_role != "Operations Manager":
    selected_model = st.sidebar.selectbox(
        "Select Model", ["XGBoost", "LightGBM", "Random Forest", "Extra Trees"]
    )
else:
    selected_model = "XGBoost"  # unused on this role's tab, kept as a default

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Role:** {user_role}")
if user_role != "Operations Manager":
    st.sidebar.markdown(f"**Model:** {selected_model}")

if all_missing:
    with st.sidebar.expander(f"⚠️ {len(all_missing)} file(s) not found", expanded=False):
        st.caption("Sections relying on these files show a placeholder instead of real data.")
        for p in all_missing:
            st.code(os.path.relpath(p, BASE_DIR), language=None)


# ─────────────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────────────
st.title("Wind Turbine Health Monitoring Dashboard")
st.markdown("Explainable AI for Early Fault Detection — Wind Farm C")
st.markdown("---")

tabs = st.tabs(visible_tabs)
tab_lookup = dict(zip(visible_tabs, tabs))


# =======================================================================
# TAB: OVERVIEW  (Control Room Operator)
# =======================================================================
if "Overview" in tab_lookup:
    with tab_lookup["Overview"]:
        st.header("Fleet Overview")

        if master_df_test is None or X_test_scaled is None or not models:
            st.warning("Fleet overview needs the test set and at least one trained model.")
        else:
            n_total_turbines_in_dataset = 22  # Wind Farm C, full dataset (all 58 labelled events)
            if len(ALL_TURBINES) < n_total_turbines_in_dataset:
                st.caption(
                    f"This view shows the **{len(ALL_TURBINES)} turbines** that have at "
                    f"least one recording in the held-out test set. Wind Farm C has "
                    f"{n_total_turbines_in_dataset} turbines in total."
                )

            model = models.get(selected_model)
            rows = []
            for asset_id in ALL_TURBINES:
                t_meta, t_scaled = get_turbine_rows(asset_id)
                if t_meta is None or len(t_meta) == 0:
                    continue
                preds = model.predict(t_scaled[feature_cols])
                non_normal_share = float(np.mean(preds != 0))
                last_pred = int(preds[-1])  # rows are chronologically sorted -> last = most recent
                rows.append({
                    "Turbine": f"Turbine {asset_id}",
                    "_asset_id": asset_id,
                    "Recordings in test set": len(t_meta),
                    "Most recent prediction": CLASS_NAMES_LIST[last_pred],
                    "Non-normal rate (%)": round(non_normal_share * 100, 1),
                    "Currently flagged": "🔴 Yes" if last_pred != 0 else "🟢 No",
                })

            if rows:
                fleet_df = pd.DataFrame(rows)

                col1, col2, col3 = st.columns(3)
                col1.metric("Turbines in test set", len(fleet_df))
                col2.metric("Currently flagged", (fleet_df["Currently flagged"] == "🔴 Yes").sum())
                col3.metric("Model in use", selected_model)

                st.markdown("##### Currently flagged turbines")
                st.caption(
                    "**Recordings in test set**: how many separate 10-minute SCADA "
                    "snapshots this turbine has in the held-out test data. "
                    "**Currently flagged**: based only on this turbine's single most "
                    f"recent recording — is {selected_model} predicting a fault right now?"
                )
                flagged_df = fleet_df[fleet_df["Currently flagged"] == "🔴 Yes"]
                if flagged_df.empty:
                    st.success("No turbines are currently flagged as faulty by their most recent recording.")
                else:
                    st.dataframe(
                        flagged_df.drop(columns=["_asset_id", "Non-normal rate (%)"]),
                        use_container_width=True, hide_index=True
                    )

                st.markdown("---")
                st.markdown("##### Most fault-prone turbines (historical rate)")
                st.caption(
                    "This is a **different statistic** from the table above: instead of "
                    "looking only at the latest recording, this looks at **all** of a "
                    "turbine's recordings in the test set and shows what percentage were "
                    "predicted as non-normal. A turbine can have a high historical rate "
                    "here while still showing 'No' above, if its most recent single "
                    "recording happened to be Normal — the two tables answer different "
                    "questions (history vs. right now)."
                )
                history_df = fleet_df.sort_values("Non-normal rate (%)", ascending=False).head(10)
                st.dataframe(
                    history_df.drop(columns=["_asset_id", "Currently flagged"]),
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning("No matching rows found for the turbines in the test set.")


# =======================================================================
# TAB: TIME SERIES  (Control Room Operator)
# =======================================================================
if "Time Series" in tab_lookup:
    with tab_lookup["Time Series"]:
        st.header("Sensor Readings Over Time")

        if master_df_test is None:
            st.warning("Time-series view needs `dashboard_data/master_df_test.csv`.")
        elif "time_stamp" not in master_df_test.columns:
            st.warning(
                "`time_stamp` column not found in `master_df_test.csv`. "
                "Add it to the export script's saved columns and re-export."
            )
        else:
            picked = turbine_picker("timeseries")
            if picked is not None:
                t_meta, t_scaled = get_turbine_rows(picked)
                if t_meta is None or len(t_meta) == 0:
                    st.warning(f"No test-set rows found for Turbine {picked}.")
                else:
                    available_cols = [c for c in feature_cols if c in t_meta.columns]

                    def label_with_unit(col):
                        name = FEATURE_NAME_MAP.get(col, col)
                        unit = None
                        if sensor_lookup is not None:
                            base_sensor = col.rsplit("_", 1)[0]
                            entry = sensor_lookup.get(base_sensor) or sensor_lookup.get(col)
                            if isinstance(entry, dict):
                                unit = entry.get("unit")
                        return f"{name} ({unit})" if unit else name

                    display_options = {label_with_unit(c): c for c in available_cols}

                    # ---- Determine which sensors to show by default: the
                    # ones actually driving this turbine's LATEST prediction,
                    # via live SHAP -- not an arbitrary/alphabetical guess.
                    # This directly answers "why would I look at this sensor?"
                    default_cols = []
                    latest_idx = len(t_meta) - 1
                    latest_row_scaled = t_scaled[feature_cols].iloc[[latest_idx]]
                    shap_vals, pred_class, _ = compute_instance_shap(selected_model, latest_row_scaled)
                    reason_text = None
                    if shap_vals is not None:
                        shap_for_pred = shap_vals[0, :, pred_class]
                        order = np.argsort(np.abs(shap_for_pred))[::-1][:4]
                        default_cols = [feature_cols[i] for i in order if feature_cols[i] in available_cols]
                        reason_text = (
                            f"Showing the sensors with the largest influence on "
                            f"**{selected_model}**'s most recent prediction for this turbine "
                            f"(**{CLASS_NAMES_LIST[pred_class]}**), computed live via SHAP — "
                            f"these are not an arbitrary or alphabetical selection."
                        )
                    else:
                        # Fallback: fleet-wide top sensors for this model, if
                        # live per-instance SHAP isn't available.
                        names, _ = compute_global_shap(selected_model)
                        if names:
                            default_cols = [c for c in available_cols if FEATURE_NAME_MAP.get(c, c) in names[:4]]
                            reason_text = (
                                f"Showing the sensors **{selected_model}** relies on most "
                                f"overall (fleet-wide), since a live explanation for this "
                                f"turbine's latest recording could not be computed."
                            )

                    default_display = [label_with_unit(c) for c in default_cols]

                    if reason_text:
                        st.info(reason_text)

                    chosen_display = st.multiselect(
                        "Sensors shown (add or remove any sensor from the full list)",
                        options=sorted(display_options.keys()),
                        default=default_display
                    )

                    st.caption(
                        "Each sensor gets its own panel below, since different sensors "
                        "use different units and scales. A dashed coloured vertical "
                        "line marks a moment this turbine was logged with a fault, "
                        "labelled with which kind — check whether the sensor's line "
                        "visibly changes around that point."
                    )

                    if chosen_display:
                        n_panels = len(chosen_display)
                        fig = make_subplots(
                            rows=n_panels, cols=1, shared_xaxes=False,
                            vertical_spacing=0.12,
                            subplot_titles=chosen_display
                        )
                        x_axis = t_meta["time_stamp"]
                        status_map = {0: "Normal", 3: "Service", 4: "Downtime", 5: "Other"}
                        non_normal = t_meta[t_meta["status_type_id"] != 0] if "status_type_id" in t_meta.columns else None

                        for row_i, disp_name in enumerate(chosen_display, start=1):
                            col = display_options[disp_name]

                            # Extract the unit (e.g. "°C", "kW", "bar") to label the
                            # y-axis honestly instead of leaving it as a bare number.
                            unit = None
                            if sensor_lookup is not None:
                                base_sensor = col.rsplit("_", 1)[0]
                                entry = sensor_lookup.get(base_sensor) or sensor_lookup.get(col)
                                if isinstance(entry, dict):
                                    unit = entry.get("unit")
                            y_label = f"Reading ({unit})" if unit else "Reading (no unit listed)"

                            fig.add_trace(
                                go.Scatter(
                                    x=x_axis, y=t_meta[col].values,
                                    mode="lines+markers", name=disp_name, showlegend=False,
                                    line=dict(color="#1f77b4")
                                ),
                                row=row_i, col=1
                            )
                            if non_normal is not None:
                                for _, fault_row in non_normal.iterrows():
                                    state_name = status_map.get(fault_row["status_type_id"], "Unknown")
                                    # Convert to a plain Python datetime: pandas Timestamp
                                    # arithmetic (used internally by Plotly's add_vline to
                                    # position its auto-annotation) raises a TypeError on
                                    # newer pandas versions, so we add the line + annotation
                                    # manually instead of via the add_vline convenience method.
                                    x_val = fault_row["time_stamp"].to_pydatetime()
                                    fig.add_shape(
                                        type="line",
                                        x0=x_val, x1=x_val, y0=0, y1=1,
                                        xref=f"x{row_i}" if row_i > 1 else "x",
                                        yref=f"y{row_i} domain" if row_i > 1 else "y domain",
                                        line=dict(
                                            dash="dash",
                                            color=CLASS_COLORS.get(state_name, "gray"),
                                            width=2.5,
                                        ),
                                        opacity=0.85,
                                    )
                                    fig.add_annotation(
                                        x=x_val, y=1, yref=f"y{row_i} domain" if row_i > 1 else "y domain",
                                        xref=f"x{row_i}" if row_i > 1 else "x",
                                        text=state_name, showarrow=False,
                                        font=dict(size=9, color=CLASS_COLORS.get(state_name, "gray")),
                                        yanchor="bottom",
                                    )

                            fig.update_xaxes(title_text="Time", row=row_i, col=1)
                            fig.update_yaxes(title_text=y_label, row=row_i, col=1)

                        fig.update_layout(
                            height=max(280, 260 * n_panels),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.caption(
                            "Each panel's time axis and zoom/pan controls are independent "
                            "of the others — moving or zooming one panel does not affect "
                            "the rest. A reading's unit (when known) is shown on its y-axis."
                        )

                        n_faults = (t_meta["status_type_id"] != 0).sum() if "status_type_id" in t_meta.columns else 0
                        if n_faults == 0:
                            st.caption("All recordings for this turbine in the test set are logged as Normal — no fault lines to show.")
                    else:
                        st.info("Select at least one sensor above to see its readings.")


# =======================================================================
# TAB: PREDICTION  (Control Room Operator + Maintenance Engineer)
# =======================================================================
if "Prediction" in tab_lookup:
    with tab_lookup["Prediction"]:
        st.header("Fault Prediction")

        if master_df_test is None or not models:
            st.warning("Prediction tab needs the test set and trained models.")
        else:
            st.caption(
                "**How this differs from Overview**: Overview only shows the *label* "
                "of the latest prediction per turbine, aggregated across all its "
                "recordings. This page lets you inspect **any single recording** in "
                "full detail — the model's confidence across all four possible "
                "states, and a direct comparison against what was actually logged "
                "for that exact moment."
            )
            picked = turbine_picker("prediction")
            if picked is not None:
                t_meta, t_scaled = get_turbine_rows(picked)
                if t_meta is None or len(t_meta) == 0:
                    st.warning(f"No test-set rows found for Turbine {picked}.")
                else:
                    n_windows = len(t_meta)
                    st.caption(
                        "Each recording is one complete 10-minute SCADA snapshot for "
                        "this turbine. Recordings are numbered in chronological order "
                        f"(recording {n_windows} = most recent — shown by default)."
                    )
                    sample_idx_1based = st.slider(
                        f"Recording number (1 to {n_windows} for Turbine {picked})",
                        min_value=1, max_value=n_windows, value=n_windows
                    )
                    sample_idx = sample_idx_1based - 1  # convert to 0-based for indexing

                    instance_scaled = t_scaled[feature_cols].iloc[[sample_idx]]
                    if "time_stamp" in t_meta.columns:
                        ts = pd.to_datetime(t_meta["time_stamp"].iloc[sample_idx])
                        st.caption(f"Timestamp: {ts}")
                    true_label_id = t_meta["status_type_id"].iloc[sample_idx] if "status_type_id" in t_meta.columns else None
                    status_map = {0: 0, 3: 1, 4: 2, 5: 3}
                    true_class = status_map.get(true_label_id) if true_label_id is not None else None

                    model = models[selected_model]
                    pred_class = int(model.predict(instance_scaled)[0])
                    pred_proba = model.predict_proba(instance_scaled)[0]

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Predicted state", CLASS_NAMES_LIST[pred_class])
                        if true_class is not None:
                            if true_class == pred_class:
                                st.success(f"✅ Matches the logged state: {CLASS_NAMES_LIST[true_class]}")
                            else:
                                st.error(
                                    f"❌ Mismatch — logged state is "
                                    f"**{CLASS_NAMES_LIST[true_class]}**, model predicted "
                                    f"**{CLASS_NAMES_LIST[pred_class]}**."
                                )
                        st.metric("Model confidence", f"{pred_proba[pred_class] * 100:.1f}%")

                    with col2:
                        fig = go.Figure(go.Bar(
                            x=CLASS_NAMES_LIST, y=pred_proba,
                            marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES_LIST],
                            text=[f"{p*100:.1f}%" for p in pred_proba], textposition="outside"
                        ))
                        fig.update_layout(
                            title=f"{selected_model} predicted class probabilities",
                            yaxis_title="Probability", yaxis_range=[0, 1], height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if true_class is not None and true_class != pred_class:
                        st.caption(
                            "A mismatch here means this specific model misclassified this "
                            "specific window. The four models in this project differ in "
                            "overall accuracy (see Fleet Analytics for the full comparison); "
                            "no model is perfect, and disagreements like this are expected "
                            "to occur at some rate even on a well-trained model."
                        )

                    if user_role == "Maintenance Engineer" and pred_class != 0:
                        st.info(
                            "This window is predicted as non-normal. Go to the "
                            "**Explanation** tab to see which sensors are driving this "
                            "prediction, including a direct comparison against normal operation."
                        )


# =======================================================================
# TAB: EXPLANATION  (Maintenance Engineer only)
# =======================================================================
if "Explanation" in tab_lookup:
    with tab_lookup["Explanation"]:
        st.header("Explainability")
        st.caption(
            "These explanations cover the three confirmed fault examples used "
            "throughout this project (one example per non-normal category: "
            "Service, Downtime, Other)."
        )

        instance_choice = st.selectbox("Fault example", ANOMALY_INSTANCE_NAMES)
        method = st.radio(
            "Explanation view",
            ["Top sensors overall", "Why this prediction (step by step)",
             "Why not Normal (comparison)", "Second opinion (LIME)",
             "Effect of one sensor (ALE)"],
            horizontal=False
        )

        slug = MODEL_SLUG[selected_model]

        instance_row = None
        true_label_for_instance = None
        asset_for_instance = None
        if anomaly_instances is not None and instance_choice in anomaly_instances:
            inst = anomaly_instances[instance_choice]
            instance_row = inst["scaled_row"][feature_cols] if feature_cols else inst["scaled_row"]
            true_label_for_instance = inst["true_label"]
            asset_for_instance = inst["asset_id"]
            st.caption(
                f"This example is Turbine {asset_for_instance}, logged as "
                f"**{CLASS_NAMES_LIST[true_label_for_instance]}**."
            )
        else:
            st.warning(
                "Anomaly instance data not found (`dashboard_data/anomaly_instances.pkl`). "
                "The views below cannot be computed without it."
            )

        # ---------------- Top sensors overall (Global SHAP) ----------------
        if method == "Top sensors overall":
            st.markdown(
                f"Across many examples, these are the sensors **{selected_model}** "
                "relies on most overall to decide a turbine's state -- not specific "
                "to the example selected above."
            )
            names, importances = compute_global_shap(selected_model)
            if names is None:
                st.warning(
                    f"Pre-computed SHAP values for {selected_model} not found "
                    f"(`{SHAP_VALUE_FILES[selected_model]}`)."
                )
            else:
                top_n = 15
                fig = go.Figure(go.Bar(
                    x=importances[:top_n][::-1], y=names[:top_n][::-1],
                    orientation="h", marker_color="#3498DB"
                ))
                fig.update_layout(
                    xaxis_title="Average influence on predictions (mean |SHAP value|)",
                    height=500, margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

        # ---------------- Local SHAP: ranked reasons for this prediction ----------------
        elif method == "Why this prediction (step by step)":
            st.markdown(
                f"This answers: **of everything the model looked at for this one "
                f"example, what mattered most in reaching its conclusion?** It does "
                f"not compare against Normal specifically (the next view does that) "
                f"-- it just ranks every sensor by how much it pushed the model "
                f"toward the state it actually predicted."
            )
            if instance_row is not None:
                shap_vals, pred_class, base_values = compute_instance_shap(selected_model, instance_row)
                if shap_vals is None:
                    st.warning(f"Model or explainer for {selected_model} not loaded.")
                else:
                    shap_for_pred = shap_vals[0, :, pred_class]
                    names = readable_names if readable_names else feature_cols
                    order = np.argsort(np.abs(shap_for_pred))[::-1][:5]

                    top_name = names[order[0]]
                    top_direction = "supported" if shap_for_pred[order[0]] > 0 else "actually argued against"
                    st.info(
                        f"📌 **In plain terms:** the single biggest factor was "
                        f"**{top_name}** — it most strongly {top_direction} the "
                        f"model's conclusion of **{CLASS_NAMES_LIST[pred_class]}** "
                        f"for this example."
                    )

                    # Relative bars only -- raw SHAP units are not on a
                    # consistent, human-meaningful scale across model types
                    # (e.g. XGBoost/LightGBM SHAP values are in raw model-margin
                    # units, not probability percentage points), so showing the
                    # literal numbers would be misleading. Bar length is scaled
                    # so the single biggest factor = 100%.
                    max_abs = np.abs(shap_for_pred[order]).max()
                    rel = (shap_for_pred[order] / max_abs) * 100
                    bar_color = ["#d62728" if v > 0 else "#1f77b4" for v in rel]
                    bar_names = [names[i] for i in order][::-1]
                    bar_vals = rel[::-1]
                    bar_color = bar_color[::-1]

                    fig = go.Figure(go.Bar(
                        x=bar_vals, y=bar_names, orientation="h", marker_color=bar_color
                    ))
                    fig.add_vline(x=0, line_color="black", line_width=1)
                    fig.update_layout(
                        title=f"{selected_model} — predicted: {CLASS_NAMES_LIST[pred_class]}",
                        xaxis_title="Relative influence (biggest factor = 100)",
                        height=380, margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Red bars supported the predicted state; blue bars actually "
                        "pointed away from it but were outweighed by the red ones. "
                        "Bar length shows relative strength only — not a percentage "
                        "of probability."
                    )

        # ---------------- Contrastive SHAP ----------------
        elif method == "Why not Normal (comparison)":
            st.markdown(
                "This answers a sharper question: **what makes this look like "
                "the predicted fault specifically — not just 'unusual', but "
                "unusual *compared to Normal operation*?** A sensor can matter a "
                "lot in the previous view, but if it pushes toward the fault AND "
                "toward Normal almost equally, it isn't actually useful for "
                "telling them apart — this view filters for sensors that do."
            )
            if instance_row is not None:
                shap_vals, pred_class, _ = compute_instance_shap(selected_model, instance_row)
                if shap_vals is None:
                    st.warning(f"Model or explainer for {selected_model} not loaded.")
                elif pred_class == 0:
                    st.info(
                        f"{selected_model} predicts **Normal** for this example, so "
                        "there is no fault state to contrast against Normal."
                    )
                else:
                    shap_A = shap_vals[0, :, pred_class]
                    shap_B = shap_vals[0, :, 0]
                    delta = shap_A - shap_B
                    names = readable_names if readable_names else feature_cols
                    top_idx = np.argsort(np.abs(delta))[-5:][::-1]

                    top_name = names[top_idx[0]]
                    st.info(
                        f"📌 **In plain terms:** **{top_name}** is the single biggest "
                        f"reason this example looks like **{CLASS_NAMES_LIST[pred_class]}** "
                        f"rather than Normal — its behaviour here differs the most "
                        f"between those two possibilities."
                    )

                    max_abs = np.abs(delta[top_idx]).max()
                    rel = (delta[top_idx] / max_abs) * 100
                    colors = ["#d62728" if v > 0 else "#1f77b4" for v in rel][::-1]
                    bar_names = [names[i] for i in top_idx][::-1]
                    bar_vals = rel[::-1]

                    fig = go.Figure(go.Bar(
                        x=bar_vals, y=bar_names, orientation="h", marker_color=colors
                    ))
                    fig.add_vline(x=0, line_color="black", line_width=1)
                    fig.update_layout(
                        title=f"{selected_model} — {CLASS_NAMES_LIST[pred_class]} vs Normal",
                        xaxis_title="Relative difference (biggest = 100)",
                        height=380, margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Red = pushes toward {CLASS_NAMES_LIST[pred_class]} more than "
                        f"toward Normal. Blue = the opposite. Bar length shows relative "
                        f"strength only."
                    )

        # ---------------- LIME (static, honest framing) ----------------
        elif method == "Second opinion (LIME)":
            st.markdown(
                "**What this is:** a completely different explanation method "
                "(LIME), used as an independent check on the SHAP views above. "
                "LIME builds a small, simplified stand-in model that only needs "
                "to be accurate for this one specific example, then shows which "
                "sensors that simplified model leaned on most. "
                "**Green bars supported the prediction. Red bars worked against it.**"
            )
            st.markdown(
                "**Why look at it:** SHAP and LIME work in totally different ways. "
                "If they point to the same sensors, that's a stronger reason to "
                "trust the explanation — it isn't just something one specific "
                "method happened to find."
            )
            st.caption(
                "Note: a separate test in this project (see Fleet Analytics) "
                "found LIME agrees with SHAP less often than SHAP agrees with "
                "itself across different models, so treat this as a helpful "
                "second check, not the main explanation."
            )
            png_path = os.path.join(BASE_DIR, f"lime_{instance_choice.lower()}_{MODEL_SLUG_SHORT[selected_model]}.png")
            if os.path.exists(png_path):
                img_col, _ = st.columns([2, 1])
                with img_col:
                    st.image(png_path, use_container_width=True)
            else:
                st.warning(f"File not found: `{os.path.basename(png_path)}`.")

        # ---------------- ALE (computed live) ----------------
        elif method == "Effect of one sensor (ALE)":
            st.markdown(
                "Shows how the model's confidence in **this prediction** would "
                "change if just *one* sensor's value were different, holding "
                "everything else about the turbine's situation realistic. "
                "Computed live across the whole fleet's test data."
            )
            names, _ = compute_global_shap(selected_model)
            if not names:
                st.warning("Could not determine the top-15 feature list for this model.")
            elif instance_row is None:
                st.warning("Select a fault example above first.")
            else:
                top15 = names[:15]
                feature_choice_name = st.selectbox("Sensor", top15)
                feature_idx_in_cols = readable_names.index(feature_choice_name) if readable_names else None
                feature_col = feature_cols[feature_idx_in_cols] if feature_idx_in_cols is not None else None

                if feature_col is None:
                    st.warning("Could not map this sensor's display name back to its data column.")
                else:
                    _, pred_class, _ = compute_instance_shap(selected_model, instance_row)
                    if pred_class is None:
                        st.warning(f"Model or explainer for {selected_model} not loaded.")
                    else:
                        bin_centers_scaled, ale_values = compute_ale_live(selected_model, feature_col, pred_class)
                        if bin_centers_scaled is None:
                            st.warning("Not enough variation in this sensor's values to compute this chart.")
                        else:
                            bin_centers_real = to_real_units(bin_centers_scaled, feature_col)

                            unit = None
                            if sensor_lookup is not None:
                                base_sensor = feature_col.rsplit("_", 1)[0]
                                entry = sensor_lookup.get(base_sensor) or sensor_lookup.get(feature_col)
                                if isinstance(entry, dict):
                                    unit = entry.get("unit")
                            x_title = f"{feature_choice_name}" + (f" ({unit})" if unit else "")

                            # Where does THIS instance's actual reading fall on the curve?
                            current_val_scaled = float(instance_row[feature_col].iloc[0])
                            current_val_real = to_real_units(np.array([current_val_scaled]), feature_col)[0]

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=bin_centers_real, y=ale_values, mode="lines",
                                line=dict(color="#2196F3", width=2.5),
                                fill="tozeroy",
                                fillcolor="rgba(31,119,180,0.25)",
                                showlegend=False
                            ))
                            fig.add_hline(y=0, line_color="black", line_width=1, line_dash="dash")
                            fig.add_vline(
                                x=current_val_real, line_color="black", line_width=2, line_dash="dot",
                                annotation_text="This example's actual reading",
                                annotation_position="top"
                            )
                            fig.update_layout(
                                xaxis_title=x_title,
                                yaxis_title=f"Effect on probability of {CLASS_NAMES_LIST[pred_class]}",
                                height=420, margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                f"Above zero = that sensor value pushes toward "
                                f"**{CLASS_NAMES_LIST[pred_class]}** more than a typical "
                                f"reading would; below zero = it pushes toward Normal. "
                                f"The dotted vertical line shows where this specific "
                                f"example's reading actually falls — if it's in a region "
                                f"clearly above zero, this sensor is genuinely "
                                f"contributing to the prediction, not just correlated "
                                f"with it by coincidence."
                            )


# =======================================================================
# TAB: FLEET ANALYTICS  (Operations Manager only)
# =======================================================================
if "Fleet Analytics" in tab_lookup:
    with tab_lookup["Fleet Analytics"]:
        st.header("Fleet Analytics")
        st.caption(
            "Aggregate, model-agnostic insights: which sensors are consistently "
            "linked to faults regardless of which model is used, and how much "
            "the different models and explanation methods agree with each other."
        )

        st.markdown("##### Top sensors agreed upon by all four models")
        st.caption(
            "First, the sensors below are picked using the combined (averaged) "
            "ranking across all four models, so every model is compared on "
            "exactly the same set of sensors. Each bar shows how important that "
            "sensor is for one specific model. Sensors with tall bars across "
            "all four models are the most trustworthy fault indicators, since "
            "they don't depend on which model is used."
        )
        per_model_importance = {}
        combined_sum = None
        ref_names = None
        for m_name in MODEL_FILES:
            arr = shap_value_arrays.get(m_name)
            if arr is None:
                continue
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))  # in feature_cols order
            per_model_importance[m_name] = mean_abs
            combined_sum = mean_abs if combined_sum is None else combined_sum + mean_abs
            ref_names = readable_names if readable_names else feature_cols

        if combined_sum is not None and len(per_model_importance) > 0:
            combined_avg = combined_sum / len(per_model_importance)
            top_idx = np.argsort(combined_avg)[::-1][:12]
            top_sensor_names = [ref_names[i] for i in top_idx]

            fig = go.Figure()
            for m_name, mean_abs in per_model_importance.items():
                vals = mean_abs[top_idx]
                vals_norm = vals / vals.max() if vals.max() > 0 else vals
                fig.add_trace(go.Bar(x=top_sensor_names, y=vals_norm, name=m_name))
            fig.update_layout(
                barmode="group", height=450,
                yaxis_title="Relative importance (normalised per model)",
                xaxis_tickangle=-40,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No SHAP value arrays found for any model.")

        st.markdown("---")
        st.markdown("##### Which turbines have actually had the most faults (historical record)")
        st.caption(
            "This shows how often a turbine was **actually logged** as faulty "
            "in the past — the real maintenance history, not a prediction. "
            "Useful for deciding which turbines need attention over the long "
            "term."
        )
        if master_df_test is not None and "asset_id" in master_df_test.columns and "status_type_id" in master_df_test.columns:
            fault_rate = (
                master_df_test.assign(is_fault=lambda d: d["status_type_id"] != 0)
                .groupby("asset_id")["is_fault"].mean()
                .sort_values(ascending=False).head(15)
            )
            fig = go.Figure(go.Bar(
                x=[f"T{a}" for a in fault_rate.index],
                y=fault_rate.values * 100, marker_color="#E74C3C"
            ))
            fig.update_layout(yaxis_title="Logged fault rate in test set (%)",
                               xaxis_title="Turbine", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Needs `master_df_test.csv` with `asset_id` and `status_type_id` columns.")