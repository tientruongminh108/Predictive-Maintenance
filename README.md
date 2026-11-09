# Predictive Maintenance: Explainable Failure-Probability Regression

A single-notebook analysis of sensor and operational data from 10 industrial machines. The notebook predicts `Failure_Prob` (0–1) from three physical sensors and, more importantly, **explains the model**: what it learned, whether the relationship is really linear, where its remaining error comes from, and what changes in the sensors would lower predicted risk.

The notebook is `predictive_maintenance.ipynb`. It runs top to bottom in about 15 seconds.

## Scope

- **Regression only.** `Maintenance_Priority` is not modeled. It is reproduced exactly (100% of rows) by two thresholds: `Downtime_Cost_USD > 4000 → 1`, else `Failure_Prob ≤ 0.40 → 3`, else `2`. A classifier would only learn that rule.
- **Inputs are the sensors** `Temp_C`, `Vibration_mm_s`, `Acoustic_dB`. Operational variables are used only as diagnostics.

## What the notebook covers

| Section | Content |
| --- | --- |
| 2–3 | Data integrity, EDA, outlier screening, why `Maintenance_Priority` is not a target |
| 4 | Train/test split and feature selection (statistical, redundancy and causal-plausibility screening, on the training split only) |
| 5 | Model selection: Linear, Ridge, Random Forest, Gradient Boosting with machine-grouped cross-validation |
| 6 | Held-out evaluation and a leave-one-machine-out check on unseen machines |
| 7.1 | Coefficients in physical units with robust confidence intervals |
| 7.2 | Global importance through three lenses: drop-one ΔR², permutation importance, mean abs SHAP |
| 7.3 | Cross-check against a black-box model: SHAP agreement, partial dependence, SHAP interactions |
| 7.4 | Residual analysis: what the sensors cannot explain, and why |
| 7.5 | Local explanations for individual readings and per-machine risk profiles |
| 7.6 | What-if analysis: sensor changes needed to reach a target risk, with feasibility limits |
| 7.7 | Influence diagnostics (Cook's distance, effect of the clipped values) |
| 8 | Conclusions and limitations |

## Results

- **Accuracy.** Linear regression on three sensors: **test R² = 0.920** (RMSE 0.0152). Predicting each machine from the other nine gives R² 0.92–0.94 per machine.
- **Drivers.** Vibration (+0.041 per mm/s) > temperature (+0.0030 per °C) > acoustic level (+0.0020 per dB). The ranking is identical under all three importance lenses, with roughly a 55% / 30% / 15% split of explained variance.
- **Additive and linear.** Gradient Boosting on the same sensors yields SHAP attributions that correlate at r ≈ 0.99 with the linear model's, straight partial-dependence curves and a lower test R² (0.904). Interaction or quadratic terms add nothing to the linear model. The temperature × vibration "interaction" suggested by `Severity_Index` is just two strong sensors bundled together.
- **Where the remaining 7% goes.** Out-of-fold residuals correlate at r = −0.98 with `Technician_Availability_pct`; adding that one variable lifts R² from 0.932 to 0.997 (RMSE within 0.0003 of the rounding floor). It is kept out of the model by design, and leaving it out does not bias the sensor coefficients (they move by less than 3%).
- **Fleet.** The 10 machines are statistically alike; per-machine sensor contributions are within sampling noise, so no machine-specific rules are warranted.

## Limitations

- The data show the hallmarks of **synthetic data** (an exactly additive target, a deterministic priority label), so the explanations describe this dataset and not real machine physics.
- Explanations and what-if analyses are **associational, not causal**.
- Ten machines, no timestamps: no assessment of drift or time-to-failure.

See Section 8 of the notebook for details.

## Repository structure

```text
predictive-maintenance/
├── predictive_maintenance.ipynb
├── smart_maintenance_dataset.csv
├── requirements.txt
└── README.md
```

## How to run

Python 3.11+ is recommended.

```bash
git clone https://github.com/tientruongminh108/predictive-maintenance.git
cd predictive-maintenance

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

jupyter lab predictive_maintenance.ipynb
```

To execute it headlessly: `jupyter nbconvert --to notebook --execute --inplace predictive_maintenance.ipynb`.

Keep `smart_maintenance_dataset.csv` next to the notebook (repository root) and run the notebook from there.

## Data and license

- **Dataset:** [Smart Manufacturing Maintenance Dataset (Kaggle)](https://www.kaggle.com/datasets/ziya07/smart-manufacturing-maintenance-dataset)
- **License:** CC0 (Public Domain)
