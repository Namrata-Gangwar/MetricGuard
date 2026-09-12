# MetricGuard — Anomaly Detection Dashboard

An interactive dashboard for detecting anomalies in time-series metrics, built on
real AWS CloudWatch data from the **Numenta Anomaly Benchmark (NAB)**.

**Live app:**[metricguard.app](https://metricguard-qbkjf8gppadcpzv6cecee8.streamlit.app)

## What it does

- Loads real-world server metrics (CPU utilization, network traffic, request counts)
- Detects anomalies using three methods: **Rolling Z-Score**, **IQR**, and **Isolation Forest**
- Scores each method's precision, recall, and F1 against NAB's labeled ground-truth anomalies
- Lets you tune thresholds live and see how detection quality changes
- Includes a "business narrative" panel explaining trade-offs between methods

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Data

Sample data is bundled in `sample_data/` — five real NAB series (EC2 CPU, EC2 network,
RDS CPU, ELB request count) plus the ground-truth label file. To add more series,
download additional CSVs from the [NAB repo](https://github.com/numenta/NAB/tree/master/data)
and drop them into `sample_data/`, then add an entry to `FRIENDLY_NAMES` and
`LABEL_KEY_PREFIX` in `app.py`.


## Citation

Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly
detection for streaming data. *Neurocomputing*.
