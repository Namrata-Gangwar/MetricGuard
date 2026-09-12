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

## Deploying for your resume/portfolio

The easiest free option is **Streamlit Community Cloud**:

1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub, and
   deploy `app.py` from that repo
3. You'll get a live public URL (e.g. `yourname-metricguard.streamlit.app`) to put
   on your resume and LinkedIn

## Talking points for interviews

- **Why three methods?** Shows you understand trade-offs between simple statistical
  rules (explainable, cheap) and ML-based detection (catches subtler patterns, harder
  to explain to non-technical stakeholders) — this is exactly the kind of "so what"
  framing a BA interviewer wants to hear.
- **Why score against labels?** Turns "I built an anomaly detector" into a
  quantified claim: "my best method achieved 0.8 F1 on real AWS metrics."
- **What would you do next?** Natural extensions: alerting thresholds, applying it
  to a business metric like SVNY sales/engagement, or a Slack/email alert integration.

## Citation

Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly
detection for streaming data. *Neurocomputing*.
