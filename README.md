# United States 2012–2016 Breast Cancer Survival Analysis Dashboard

Interactive **Streamlit** app to explore breast cancer survival using SEER (2012–2016).

- **Time Since Diagnosis** — Kaplan–Meier curves with 95% CIs
- **Recent Trend** — Year-of-diagnosis (YearDx) **12-month survival** trends, Δ vs previous year (▲/▼), and headline metrics

## Table of Contents

- [Features](#features)
- [Data](#data)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Acknowledgements](#Acknowledgements)

## Features

- Compare by selections: **Race/Ethnicity**, **Age group**, **Stage**, **Grade**
- Sidebar filters: **Race/Ethnicity**, **Age group**, **Stage**, **Grade**, **Histology**
- **Kaplan–Meier** curves per group with shaded 95% CI bands + chisq test on survival months
- **Trend view**: recent years 1-year survival lines with CI bands + chisq test on 1-year survival status

## Data

SEER website: https://seer.cancer.gov/

They collected the data based on End Results data from a series of hospital registries and one population-based registry.

> ⚠️ Follow SEER data-use terms; do **not** commit PHI/PII.

## Deployed App

Link: https://nannmemeda-survival-analysis-dashboard-app-8dcvhm.streamlit.app/

## Quick Start

If you want to run the dashboard in your own PC, please follow this instruction.

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Upgrade pip & install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## Project Structure

```bash
├─ app.py                 # Streamlit app
├─ cleaned_SEER.csv
├─ data_clean_exploration.ipynb                 # Data clean process
├─ requirements.txt
└─ README.md
```

## Acknowledgements
Built with streamlit, plotly, scipy, pandas, numpy and lifelines
