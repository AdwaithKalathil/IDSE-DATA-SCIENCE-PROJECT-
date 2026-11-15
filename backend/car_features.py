"""
car_features.py
---------------
Shared code for the used-car price project.

Everything that turns RAW user inputs into model-ready numbers lives here, and is
wrapped in scikit-learn objects so it can be saved INSIDE best_model.pkl.
That way the Tkinter frontend only ever passes raw values (year, fuel, brand, ...).

IMPORTANT: best_model.pkl references the class `CarFeatureEngineer` below, so this
file must be importable (same folder, or on PYTHONPATH) whenever the model is loaded.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
REFERENCE_YEAR = 2026          # car age = REFERENCE_YEAR - manufacturing year

# ---- The exact raw fields the saved pipeline expects --------------------------
RAW_NUMERIC = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
RAW_CATEGORICAL = ["fuel", "seller_type", "transmission", "owner", "brand"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# owner is ORDERED (more previous owners -> usually lower price), so we encode 1..4
OWNER_ORDER = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3,
               "Fourth & Above Owner": 4}

# Brand tier = market positioning (domain knowledge), NOT computed from prices in the
# data, so it cannot leak the target. Unknown/rare brands default to "mid".
BRAND_TIER = {
    # budget / mass-market
    "Maruti": "budget", "Datsun": "budget", "Daewoo": "budget", "Fiat": "budget",
    "Chevrolet": "budget", "Tata": "budget", "Renault": "budget",
    "Ambassador": "budget", "Force": "budget", "Ashok": "budget", "Opel": "budget",
    # mid-range
    "Hyundai": "mid", "Honda": "mid", "Toyota": "mid", "Ford": "mid",
    "Mahindra": "mid", "Volkswagen": "mid", "Skoda": "mid", "Nissan": "mid",
    "Mitsubishi": "mid", "Isuzu": "mid", "Kia": "mid", "MG": "mid", "Jeep": "mid",
    "Peugeot": "mid",
    # luxury
    "BMW": "luxury", "Mercedes-Benz": "luxury", "Audi": "luxury", "Jaguar": "luxury",
    "Land Rover": "luxury", "Volvo": "luxury", "Lexus": "luxury",
}


def parse_number(value):
    """'23.4 kmpl' -> 23.4 ; '1248 CC' -> 1248.0 ; '74 bhp' -> 74.0 ; junk/NaN -> NaN."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else np.nan


def extract_brand(name):
    """First word of the car name; 'Land Rover' is two words so it is special-cased."""
    first = str(name).strip().split()[0]
    return "Land Rover" if first == "Land" else first


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw columns -> engineered columns. Stateless (nothing is learned from data),
    so it can never leak information between train and test."""

    ENGINEERED = ["age", "km_driven", "km_per_year", "mileage", "engine", "max_power",
                  "seats", "owner_num", "fuel", "seller_type", "transmission",
                  "brand", "brand_tier"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        out["age"] = REFERENCE_YEAR - pd.to_numeric(X["year"], errors="coerce")
        out["km_driven"] = pd.to_numeric(X["km_driven"], errors="coerce")
        out["km_per_year"] = out["km_driven"] / out["age"].clip(lower=1)
        for c in ["mileage", "engine", "max_power", "seats"]:
            out[c] = pd.to_numeric(X[c], errors="coerce")
        out["owner_num"] = X["owner"].map(OWNER_ORDER)      # unknown -> NaN -> imputed
        for c in ["fuel", "seller_type", "transmission", "brand"]:
            out[c] = X[c].astype(str)
        out["brand_tier"] = X["brand"].map(BRAND_TIER).fillna("mid")
        return out[self.ENGINEERED]


NUM_COLS = ["age", "km_driven", "km_per_year", "mileage", "engine", "max_power",
            "seats", "owner_num"]
CAT_COLS = ["fuel", "seller_type", "transmission", "brand", "brand_tier"]


def build_preprocessor():
    """Imputation / scaling / one-hot live in the pipeline, so they are fitted on the
    TRAINING fold only and merely applied to validation / test / new cars."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                        sparse_output=False)
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def build_pipeline(regressor, log_target=True):
    """raw DataFrame -> feature engineering -> preprocessing -> model -> price in INR.
    With log_target=True the model is trained on log(1+price) and predictions are
    transformed back to rupees automatically."""
    if log_target:
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1,
            check_inverse=False)
    return Pipeline([("features", CarFeatureEngineer()),
                     ("prep", build_preprocessor()),
                     ("model", regressor)])
