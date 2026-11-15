"""
predict.py — what the Tkinter frontend calls.

    from predict import predict_price, predict_with_range
    price = predict_price({...raw car fields...})          # float, INR
    info  = predict_with_range({...})                       # dict with low/high + warnings

Pass RAW values only (the saved pipeline does the encoding / imputation / log-transform).
Required fields : year, km_driven, fuel, seller_type, transmission, owner, brand
Optional fields : mileage, engine, max_power, seats   (None / omitted -> median-imputed)
Allowed category values and training ranges are in model_input_schema.json.

Files needed side by side: predict.py, car_features.py, best_model.pkl, model_input_schema.json
"""
import json
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                     # so best_model.pkl can import car_features
    sys.path.insert(0, HERE)

import car_features as cf                    # noqa: E402  (must be imported before joblib.load)

_MODEL = joblib.load(os.path.join(HERE, "best_model.pkl"))
with open(os.path.join(HERE, "model_input_schema.json"), encoding="utf-8") as f:
    SCHEMA = json.load(f)

REQUIRED = [k for k, v in SCHEMA["input_fields"].items() if v["required"]]
OPTIONAL = [k for k, v in SCHEMA["input_fields"].items() if not v["required"]]


def _to_frame(car: dict) -> pd.DataFrame:
    missing = [k for k in REQUIRED if car.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")
    for k, spec in SCHEMA["input_fields"].items():          # categorical values must be known
        val = car.get(k)
        if "allowed_values" in spec and val is not None and k != "brand" and val not in spec["allowed_values"]:
            raise ValueError(f"'{val}' is not a valid {k}. Allowed: {spec['allowed_values']}")
    row = {k: car.get(k) for k in cf.RAW_FEATURES}           # exact column set + order
    for k in ["year", "km_driven", "mileage", "engine", "max_power", "seats"]:
        row[k] = pd.to_numeric(row[k]) if row[k] is not None else float("nan")
    return pd.DataFrame([row], columns=cf.RAW_FEATURES)


def _warnings(car: dict) -> list:
    w = []
    f = SCHEMA["input_fields"]
    if not f["year"]["train_min"] <= int(car["year"]) <= f["year"]["train_max"]:
        w.append(f"Year {car['year']} is outside the training range "
                 f"{f['year']['train_min']}-{f['year']['train_max']}; the estimate is unreliable.")
    if car["brand"] not in f["brand"]["allowed_values"]:
        w.append(f"Brand '{car['brand']}' was not seen in training; estimate relies on other specs.")
    elif SCHEMA["brand_training_counts"].get(car["brand"], 0) < 30:
        w.append(f"Only {SCHEMA['brand_training_counts'][car['brand']]} '{car['brand']}' cars in training; low confidence.")
    return w


def predict_price(car: dict) -> float:
    """Predicted resale price in INR for ONE car given as a dict of raw inputs."""
    return float(_MODEL.predict(_to_frame(car))[0])


def predict_with_range(car: dict) -> dict:
    """Point estimate + approx. 80% range (empirical, from test-set errors) + warnings."""
    p = predict_price(car)
    band = SCHEMA["error_band"]
    return {"price": round(p), "low": round(p * band["low_factor"]),
            "high": round(p * band["high_factor"]), "warnings": _warnings(car)}


if __name__ == "__main__":
    example = {"year": 2015, "km_driven": 60000, "fuel": "Diesel", "seller_type": "Individual",
               "transmission": "Manual", "owner": "First Owner", "mileage": 23.4,
               "engine": 1248, "max_power": 74, "seats": 5, "brand": "Maruti"}
    r = predict_with_range(example)
    print(f"Estimated price: ₹{r['price']:,}   (likely range ₹{r['low']:,} – ₹{r['high']:,})")
    for msg in r["warnings"]:
        print("Warning:", msg)
