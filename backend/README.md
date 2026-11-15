# Used-Car Resale Price Prediction (CarDekho, India)

Introduction to Data Science — course project. Python, scikit-learn, XGBoost, SHAP. Random seed **42** everywhere.

## 1. Problem statement
**Core task:** predict the resale price (INR) of a used car from its listing details (year, km driven, fuel, seller type, transmission, owner count, mileage, engine, max power, seats, brand).
**Secondary analyses:** (a) which factors drive resale value (SHAP + permutation importance); (b) how predicted price trends with car age across brands and segments (budget / mid / luxury).

## 2. Data source
*Vehicle dataset from CarDekho* (`Car details v3.csv`), Nehal Birla, Kaggle —
<https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>. Listings scraped from the Indian used-car marketplace CarDekho.
8,128 raw rows × 13 columns → **6,923 rows** after cleaning (1,202 exact duplicates and 3 impossible odometer readings removed). The CSV is included in `data/`.

## 3. Headline results (held-out test set, 693 cars, models re-fitted on train+validation)
| Model | RMSE (₹) | MAE (₹) | R² | MAPE (%) | RMSLE |
|---|---:|---:|---:|---:|---:|
| Baseline (median price) | 529,688 | 278,974 | −0.051 | 80.8 | 0.772 |
| Linear Regression | 185,391 | 97,595 | 0.871 | 20.8 | 0.255 |
| Random Forest (tuned) | 153,497 | 76,336 | 0.912 | 16.7 | 0.216 |
| **XGBoost (tuned) — saved model** | 156,309 | **74,713** | 0.908 | **16.4** | **0.215** |

Bootstrap 95% CI for XGBoost: MAE ₹65k–₹86k, MAPE 15.1–17.8%, R² 0.85–0.95. Random Forest and XGBoost are statistically tied on the test set (MAE-gap CI includes 0; each wins on different metrics); XGBoost was chosen because it won on validation on every metric, decided before looking at the test set.
**Main drivers (SHAP):** car age ≫ max power > brand > engine size > fuel; km driven, owner count, transmission, seller type matter little once age is accounted for. Cars lose roughly 10% of value per year (budget/mid) and ~13% per year (luxury, −12.8%).

## 4. Project structure
```
car_price_project/
├── README.md                     <- this file
├── requirements.txt
├── car_price_prediction.ipynb    <- full analysis, executed end to end (outputs included)
├── car_features.py               <- cleaning helpers + feature engineering + sklearn pipeline builder
├── predict.py                    <- load saved pipeline, predict one car from RAW inputs (frontend entry point)
├── best_model.pkl                <- fitted full pipeline (feature engineering -> encoding -> XGBoost -> price in INR)
├── model_input_schema.json       <- exact raw input fields, dtypes, allowed values, training ranges
├── data/Car_details_v3.csv       <- raw dataset
├── figures/                      <- 12 PNG plots used in the notebook and report
└── results/                      <- metric tables (CSV), cleaning log, SHAP/permutation importance, what-if effects
```

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A: open and run interactively
jupyter notebook car_price_prediction.ipynb            # Run > Run All Cells

# Option B: run end to end from the command line (~2-3 minutes on one CPU core)
jupyter nbconvert --to notebook --execute --inplace car_price_prediction.ipynb

# Try a prediction
python predict.py
```
Running the notebook regenerates `best_model.pkl`, `model_input_schema.json`, all figures and all result tables. With the pinned versions and `random_state=42`, results are identical on re-runs.

## 6. Method in brief
- **Cleaning** (reasoning for each step is in the notebook, §2): regex-parse `mileage`/`engine`/`max_power`; `max_power = 0` and `mileage = 0` → missing; drop messy `torque`; remove duplicates *before* splitting; remove impossible `km_driven`; merge the 5-row "Test Drive Car" owner level; extract `brand`.
- **Features:** `age = 2026 − year`, `km_per_year`, `brand_tier` (budget/mid/luxury by brand positioning, not computed from prices), ordinal `owner_num`.
- **Split:** 80 / 10 / 10 train / validation / test (`random_state=42`, stratified on price deciles). EDA on the training set only.
- **No leakage:** imputation, scaling and one-hot encoding are inside the sklearn `Pipeline` (re-fitted per CV fold); engineered features are stateless.
- **Models:** median baseline, Linear Regression, Random Forest, XGBoost. Target = log(price) via `TransformedTargetRegressor` (pipeline still outputs INR). RF and XGBoost tuned with `RandomizedSearchCV`, 5-fold CV on the training set, scored on RMSLE. Winner picked on validation; final scores on the untouched test set.
- **Metrics:** RMSE, MAE, R², MAPE (+ RMSLE). See the notebook for why each one is used.

## 7. Limitations (details in notebook §8 and the report)
1. Snapshot of *asking* prices (~2020), nominal INR; data only up to model year 2020 and tree models cannot extrapolate beyond it.
2. No condition / accident / service history.
3. Uneven brand coverage (Maruti + Hyundai ≈ 49%; luxury ≈ 2%); luxury and rare-brand estimates are less reliable.
4. Brand only (no model/variant); small test set → wide confidence intervals.

**Next steps:** (1) add model/variant (+ listing date / condition if available); (2) give calibrated price ranges via quantile regression or conformal prediction.

---

## 8. For the Tkinter frontend (separate session)
Copy these four files into the frontend folder: **`predict.py`, `car_features.py`, `best_model.pkl`, `model_input_schema.json`** (`best_model.pkl` imports `car_features.py`, so it must sit next to them). Install the pinned `scikit-learn` / `xgboost` versions from `requirements.txt`.

```python
from predict import predict_price, predict_with_range

car = {"year": 2015, "km_driven": 60000, "fuel": "Diesel", "seller_type": "Individual",
       "transmission": "Manual", "owner": "First Owner", "mileage": 23.4,
       "engine": 1248, "max_power": 74, "seats": 5, "brand": "Maruti"}

predict_price(car)          # -> 531480.0   (float, INR)
predict_with_range(car)     # -> {'price': 531480, 'low': 413492, 'high': 663287, 'warnings': []}
```
Pass **raw** values only. All encoding, imputation, age / km-per-year / brand-tier calculation and the inverse log-transform happen inside the saved pipeline.

### Raw input fields the pipeline expects
| Field | dtype | Required | Allowed values / range (training data) |
|---|---|---|---|
| `year` | int | yes | manufacturing year, **1983–2020** (outside = extrapolation; `predict_with_range` warns) |
| `km_driven` | int | yes | 1,000 – 577,414 |
| `fuel` | str | yes | `CNG`, `Diesel`, `LPG`, `Petrol` |
| `seller_type` | str | yes | `Dealer`, `Individual`, `Trustmark Dealer` |
| `transmission` | str | yes | `Automatic`, `Manual` |
| `owner` | str | yes | `First Owner`, `Second Owner`, `Third Owner`, `Fourth & Above Owner` |
| `brand` | str | yes | 32 brands, see `model_input_schema.json` (e.g. `Maruti`, `Hyundai`, `BMW`, `Mercedes-Benz`, `Land Rover`); unseen brands are tolerated (a warning is returned) |
| `mileage` | float | no | kmpl (km/kg for CNG/LPG); `None` → median-imputed |
| `engine` | float | no | CC (624–3,604 typical); `None` → median-imputed |
| `max_power` | float | no | bhp; `None` → median-imputed |
| `seats` | float | no | 2–14; `None` → median-imputed |

Frontend tips: use dropdowns (Combobox) filled from `model_input_schema.json["input_fields"][field]["allowed_values"]`; cap the year spinbox at 2020; show the `low`–`high` range and any `warnings`; remember that predictions are 2020-era asking prices in nominal INR and are not reliable for cars younger than the training data.
