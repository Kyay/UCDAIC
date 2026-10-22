# AIC Flight Delay Predictor

An end-to-end Machine Learning pipeline and predictive application developed for the **AI Collective (AIC) @ UC Davis** (Fall 2026 Beginner AI Project Cycle).

This project uses flight telemetry data to analyze airline schedule reliability, predict continuous arrival delay lengths, and classify flight delay probabilities ($\ge 15$ minutes) using Federal Aviation Administration (FAA) criteria.

---

## Repository Structure

```text
.
├── Readme.md
├── data/
│   └── flights_sample.parquet      # Flight operations telemetry dataset
└── week1/
    ├── requirement.txt             # Environment dependencies for Week 1
    └── week_material.ipynb         # EDA, data cleaning, and baseline regression notebook
```

---

## Project Overview

Flight delays disrupt travel schedules and create logistical bottlenecks across airport hubs. This project builds a complete machine learning workflow from data cleaning to model deployment:

* **Dataset:** Serialized Apache Parquet dataset (`data/flights_sample.parquet`) optimized for memory efficiency and fast column reads.
* **Task 1 (Regression - Week 1):** Predict exact arrival delay duration in minutes ($y \in \mathbb{R}$) using flight distance, elapsed time, and departure metrics.
* **Task 2 (Classification - Week 2):** Predict binary flight delay status ($y \in \{0, 1\}$, where $1$ denotes an arrival delay $\ge 15$ minutes).
* **Delivery (Weeks 5 & 6):** Export a serialized scikit-learn pipeline using `joblib`, serve predictions via a FastAPI endpoint, and connect to an interactive frontend UI.

---

## 6-Week Project Roadmap

| Phase | Focus & Milestones | Core Deliverables |
| :--- | :--- | :--- |
| **Week 1** | **Data Preparation & Baseline Regression** | Load Parquet data, handle missing values, run EDA with Pandas/NumPy, fit baseline `LinearRegression` / Polynomial models, evaluate with $R^2$ and MSE. |
| **Week 2** | **Feature Engineering & Classification** | One-hot encode categorical features (carriers, origins), compute delay-to-flight interaction ratios, train Logistic Regression and Random Forest models. |
| **Week 3** | **Model Evaluation & Visualization** | $K$-fold cross-validation, Seaborn confusion matrix heatmaps, ROC-AUC curves, precision-recall trade-offs. |
| **Week 4** | **Hyperparameter Optimization** | Systematically tune tree estimators, max depth, and regularization using `GridSearchCV` / `RandomizedSearchCV`. |
| **Week 5** | **Model Serialization & API** | Serialize the scikit-learn pipeline to disk using `joblib`, build a REST inference API with FastAPI. |
| **Week 6** | **UI Deployment & Showcase Pitch** | Connect interactive UI (Streamlit or web client) to the FastAPI backend, deliver live Shark Tank-style project pitch (Dec 2). |

---

## Team Roles & Ownership

* **Product Manager (PM):** Scope management, timeline tracking, milestone reviews, slide deck preparation, and final pitch lead.
* **Developer 1 (Data & Preprocessing Lead):** Parquet ingestion, missing value imputation, EDA visualizations, feature transformations (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
* **Developer 2 (Machine Learning Lead):** Baseline regression, classification algorithms, metric evaluation (ROC-AUC, F1-score), hyperparameter tuning (`GridSearchCV`).
* **Developer 3 (Full-Stack & Integration Lead):** FastAPI backend architecture, API contract schema, UI development, model serving and integration.

---

## Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/milesmartinez423/AIC-Flight-Project.git
cd AIC-Flight-Project
```

### 2. Set Up a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r week1/requirement.txt
```

> **Note on Parquet Files:** Reading `.parquet` files via Pandas requires an engine such as `pyarrow`. If not listed in your requirements, install it with:
> ```bash
> pip install pyarrow
> ```

### 4. Reading the Parquet Dataset
In Python or inside `week_material.ipynb`:
```python
import pandas as pd

# Load flight parquet records
df = pd.read_parquet("data/flights_sample.parquet")
print("Shape:", df.shape)
print(df.head())
```

### 5. Launch the Starter Notebook
```bash
jupyter notebook week1/week_material.ipynb
```

---

## Git Workflow Guidelines

To maintain clean repository hygiene across team members:

1. **Never commit directly to `main`:** All work must be done on isolated branches.
2. **Branch Naming Standard:**
   * `feature/week1-baseline-regression`
   * `feature/week2-classification`
   * `feature/backend-api`
3. **Branch Creation & Updates:**
   ```bash
   git switch main
   git pull origin main
   git switch -c feature/<your-feature-name>
   ```
4. **Committing Changes:**
   ```bash
   git add <modified-files>
   git commit -m "feat(week1): add baseline linear regression model"
   git push origin feature/<your-feature-name>
   ```
5. **Pull Requests:** Open a PR on GitHub for team review before merging into `main`.

---

## License & Acknowledgments

Developed as part of the **AI Collective at UC Davis** Beginner AI Project Track (Fall 2026).