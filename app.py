"""
MetricGuard — Anomaly Detection Dashboard
Built on the Numenta Anomaly Benchmark (NAB) real-world AWS CloudWatch metrics.
Run with: streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="MetricGuard", layout="wide", page_icon="📈")

DATA_DIR = "sample_data"
LABELS_PATH = os.path.join(DATA_DIR, "combined_labels.json")

FRIENDLY_NAMES = {
    "ec2_cpu_utilization_24ae8d.csv": "EC2 CPU Utilization (Instance A)",
    "ec2_cpu_utilization_825cc2.csv": "EC2 CPU Utilization (Instance B)",
    "ec2_network_in_5abac7.csv": "EC2 Network In (bytes)",
    "rds_cpu_utilization_cc0c53.csv": "RDS CPU Utilization",
    "elb_request_count_8c0756.csv": "ELB Request Count",
}

LABEL_KEY_PREFIX = {
    "ec2_cpu_utilization_24ae8d.csv": "realAWSCloudwatch/",
    "ec2_cpu_utilization_825cc2.csv": "realAWSCloudwatch/",
    "ec2_network_in_5abac7.csv": "realAWSCloudwatch/",
    "rds_cpu_utilization_cc0c53.csv": "realAWSCloudwatch/",
    "elb_request_count_8c0756.csv": "realAWSCloudwatch/",
}


@st.cache_data
def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_series(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_true_anomaly_timestamps(filename, labels):
    key = LABEL_KEY_PREFIX.get(filename, "") + filename
    return pd.to_datetime(labels.get(key, []))


def zscore_detector(df, window, threshold):
    roll_mean = df["value"].rolling(window, min_periods=1, center=True).mean()
    roll_std = df["value"].rolling(window, min_periods=1, center=True).std().replace(0, np.nan)
    z = (df["value"] - roll_mean) / roll_std
    return z.abs() > threshold


def iqr_detector(df, window, multiplier):
    q1 = df["value"].rolling(window, min_periods=1, center=True).quantile(0.25)
    q3 = df["value"].rolling(window, min_periods=1, center=True).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (df["value"] < lower) | (df["value"] > upper)


def isolation_forest_detector(df, contamination):
    roll_mean = df["value"].rolling(20, min_periods=1).mean()
    roll_std = df["value"].rolling(20, min_periods=1).std().fillna(0)
    diff = df["value"].diff().fillna(0)
    features = np.column_stack([df["value"], roll_mean, roll_std, diff])
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(features)
    return pd.Series(preds == -1, index=df.index)


def score_against_labels(pred_flags, timestamps, true_timestamps, tolerance_minutes=60):
    """A predicted flag counts as a true positive if it falls within
    `tolerance_minutes` of any true anomaly timestamp."""
    if len(true_timestamps) == 0:
        return None
    pred_times = timestamps[pred_flags]
    tol = pd.Timedelta(minutes=tolerance_minutes)

    tp = 0
    matched_true = set()
    for pt in pred_times:
        diffs = (true_timestamps - pt).total_seconds().__abs__()
        idx_min = diffs.argmin() if len(diffs) else None
        if idx_min is not None and diffs[idx_min] <= tol.total_seconds():
            tp += 1
            matched_true.add(idx_min)

    fp = len(pred_times) - tp
    fn = len(true_timestamps) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------- Sidebar controls ----------------
st.sidebar.title("⚙️ Controls")

available_files = [f for f in FRIENDLY_NAMES if os.path.exists(os.path.join(DATA_DIR, f))]
selected_file = st.sidebar.selectbox(
    "Metric (NAB dataset)",
    available_files,
    format_func=lambda f: FRIENDLY_NAMES.get(f, f),
)

method = st.sidebar.radio(
    "Detection method",
    ["Rolling Z-Score", "IQR", "Isolation Forest", "Compare all three"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

window = st.sidebar.slider("Rolling window (points)", 10, 200, 50, step=10)
z_threshold = st.sidebar.slider("Z-score threshold", 1.5, 5.0, 3.0, step=0.1)
iqr_multiplier = st.sidebar.slider("IQR multiplier", 1.0, 3.0, 1.5, step=0.1)
contamination = st.sidebar.slider("Isolation Forest: expected anomaly %", 0.01, 0.15, 0.03, step=0.01)

tolerance = st.sidebar.slider(
    "Scoring tolerance (minutes)",
    5, 180, 60, step=5,
    help="A detected point counts as correct if it falls within this many minutes of a labeled anomaly.",
)

# ---------------- Load data ----------------
df = load_series(selected_file)
labels = load_labels()
true_anomalies = get_true_anomaly_timestamps(selected_file, labels)

# ---------------- Header ----------------
st.title("📈 MetricGuard")
st.caption(
    "Anomaly detection dashboard built on real AWS CloudWatch metrics from the "
    "Numenta Anomaly Benchmark (NAB)."
)

col1, col2, col3 = st.columns(3)
col1.metric("Data points", f"{len(df):,}")
col2.metric("Labeled anomalies", len(true_anomalies))
col3.metric(
    "Time range",
    f"{(df['timestamp'].max() - df['timestamp'].min()).days} days",
)

# ---------------- Run detectors ----------------
methods_to_run = (
    ["Rolling Z-Score", "IQR", "Isolation Forest"] if method == "Compare all three" else [method]
)

results = {}
for m in methods_to_run:
    if m == "Rolling Z-Score":
        flags = zscore_detector(df, window, z_threshold)
    elif m == "IQR":
        flags = iqr_detector(df, window, iqr_multiplier)
    else:
        flags = isolation_forest_detector(df, contamination)
    results[m] = flags

# ---------------- Chart ----------------
colors = {"Rolling Z-Score": "#EF553B", "IQR": "#AB63FA", "Isolation Forest": "#FFA15A"}

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines", name="Metric", line=dict(color="#636EFA", width=1.5)))

for m, flags in results.items():
    flagged = df[flags]
    fig.add_trace(
        go.Scatter(
            x=flagged["timestamp"], y=flagged["value"], mode="markers",
            name=f"{m} flags", marker=dict(color=colors.get(m), size=8, symbol="x"),
        )
    )

for i, t in enumerate(true_anomalies):
    fig.add_vline(x=t, line_dash="dot", line_color="green", opacity=0.5,
                   annotation_text="true anomaly" if i == 0 else None)

fig.update_layout(
    height=500, margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Time", yaxis_title="Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Scoring ----------------
st.subheader("📊 Detection performance vs. labeled ground truth")

score_rows = []
for m, flags in results.items():
    scores = score_against_labels(flags, df["timestamp"], true_anomalies, tolerance)
    if scores:
        score_rows.append(
            {
                "Method": m,
                "Flagged points": int(flags.sum()),
                "Precision": round(scores["precision"], 2),
                "Recall": round(scores["recall"], 2),
                "F1": round(scores["f1"], 2),
                "True positives": scores["tp"],
                "False positives": scores["fp"],
                "False negatives": scores["fn"],
            }
        )

if score_rows:
    score_df = pd.DataFrame(score_rows)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    if len(score_rows) > 1:
        best = score_df.loc[score_df["F1"].idxmax(), "Method"]
        st.success(f"**Best performer on this series:** {best} (highest F1 score)")
else:
    st.info("No labeled anomalies available for this series to score against.")

# ---------------- Business narrative ----------------
with st.expander("💡 How to read this for a business audience"):
    st.markdown(
        """
        - **Rolling Z-Score** is cheap to compute and easy to explain to stakeholders
          ("flag anything 3 standard deviations from the recent average") — good for
          real-time monitoring where interpretability matters.
        - **IQR** is more robust to skewed distributions (e.g. traffic spikes that
          aren't symmetric) and doesn't assume a normal distribution.
        - **Isolation Forest** can catch subtler, multivariate anomalies (e.g. a value
          that's normal on its own but unusual given the recent trend), at the cost of
          being harder to explain in a one-line business rule.
        - In a real monitoring setup, you'd typically start with Z-Score/IQR for
          cheap, explainable alerts, and add Isolation Forest for a second-pass check
          on high-value metrics.
        """
    )

st.caption(
    "Data: Numenta Anomaly Benchmark (NAB), real AWS CloudWatch metrics. "
    "https://github.com/numenta/NAB"
)
