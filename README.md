# Predictive Maintenance Analysis

This project implements a predictive maintenance analysis workflow on sensor and operational data collected from 10 industrial machines.

The solution covers two modeling tasks:
* **Regression:** Predict physical `Failure_Prob` (0–1) from real-time sensor metrics.
* **Classification:** Predict `Maintenance_Priority` (High / Medium / Low) to support Multi-Criteria Decision-Making (MCDM).

## Contents
Notebook	| Purpose |
--- | --- |
`01_eda_preprocessing.ipynb` | Data integrity checks, univariate/bivariate analysis, outlier handling, feature engineering (`Severity_Index`, `Resource_Risk`), correlation analysis. Outputs `processed_dataset.csv`. |
`02_regression_model.ipynb`	| Compares Linear Regression, Random Forest, Gradient Boosting under one consistent, machine-grouped CV methodology; selects and tunes the winner. |
`03_classification_model.ipynb` | Evaluates a two-stage pipeline (auxiliary regression $\rightarrow$ classification); compares Logistic Regression, Random Forest, and CatBoost; tunes hyperparameters with Optuna; includes SHAP-based error analysis. |

## Results

- **Regression** - Linear Regression achieved **Test $R^2$ = 0.920** using only three physical sensor features (`Temp_C`, `Vibration_mm_s`, `Acoustic_dB`). Features lacking a physical connection to machine failure (`Technician_Availability_pct`, `Downtime_Cost_USD`, `Inspection_Duration_min`) were excluded despite raising $R^2$ to 0.997, as their predictive power stemmed from target generation artifacts rather than genuine physical signals.

- **Classification:** CatBoost achieved **Test Macro F1 = 0.958** (Priority 1: 1.000, Priority 2: 0.990, Priority 3: 0.882). To prevent data leakage and simulate true inference conditions, the model uses out-of-fold predictions from the auxiliary regression stage instead of historical ground-truth `Failure_Prob`.

## Key Insights

* **Rule-Driven Critical Priorities:** Priority 1 (urgent) decisions are dominated by `Downtime_Cost_USD` crossing a ~$4,000 threshold. This suggests Priority 1 is partially rule-derived in the underlying data generation process and should be verified with domain experts before operational rollout.
* **True Signal in Operational Triage:** The distinction between Priority 2 and Priority 3 reflects genuine physical sensor patterns rather than operational cost shortcuts. Evaluated in isolation, the model maintains strong discriminatory power (Macro F1 $\approx$ 0.93).
* **Linearity of Physical Failure:** Machine failure probability behaves close to a linear function of physical sensor readings, dominated by a temperature $\times$ vibration interaction. Simple linear models match or outperform complex tree ensembles on this task.
* **Causally Sound Modeling:** Non-physical operational features improve raw statistical metrics but lack causal validity for predicting mechanical breakdown. Feature selection across all tasks strictly prioritized causal justification over metric-chasing.

## Repo Structure

```text
predictive_maintenance/
│
├── data/
│   ├── raw/             
│   └── processed/    
│
├── models/                 
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_regression_model.ipynb
│   └── 03_classification_model.ipynb     
│
├── src/                 
│   └── utils.py  
│
├── .gitignore            
├── requirements.txt  
└── README.md        
```
## How to Run

Ensure you have **Python 3.8+** installed. Clone the repository and install the required dependencies:

```bash
# Clone the repo
git clone https://github.com/tientruongminh108/predictive-maintenance.git
cd predictive-maintenance

# Create and activate a virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate  # On MacOS / Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Execution Workflow
Run the Jupyter notebooks in sequential order:
1. `notebooks/01_eda_preprocessing.ipynb` - clean raw data and construct engineered features.
2. `notebooks/02_regression_model.ipynb` - train the baseline physical regression model.
3. `notebooks/03_classification_model.ipynb` - run the auxiliary two-stage classification stage.

**Note**: Pipeline utilities in `src/utils.py` handle cross-notebook predictions and ensure out-of-fold alignment across independent data splits.

## Data & License

* **Dataset:** [Smart Manufacturing Maintenance Dataset - Kaggle](https://www.kaggle.com/datasets/ziya07/smart-manufacturing-maintenance-dataset)
* **License:** CC0 (Public Domain)