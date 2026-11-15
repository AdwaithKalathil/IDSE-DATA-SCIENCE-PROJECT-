"""Small Flask API for the used-car resale price frontend."""
from __future__ import annotations

import json
import math
import os
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SCHEMA_PATH = os.path.join(BASE_DIR, "model_input_schema.json")

with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
    SCHEMA = json.load(schema_file)

FIELDS = SCHEMA["input_fields"]
REQUIRED_FIELDS = [name for name, spec in FIELDS.items() if spec.get("required")]
OPTIONAL_FIELDS = [name for name, spec in FIELDS.items() if not spec.get("required")]
NUMERIC_FIELDS = {name for name, spec in FIELDS.items() if spec.get("dtype") in {"int", "float"}}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _normalise_and_validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object containing the car fields.")

    allowed_keys = set(FIELDS)
    extra = sorted(set(payload) - allowed_keys)
    if extra:
        raise ValueError(f"Unknown field(s): {', '.join(extra)}.")

    missing = [name for name in REQUIRED_FIELDS if payload.get(name) is None or payload.get(name) == ""]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}.")

    car: dict[str, Any] = {}
    for name, spec in FIELDS.items():
        value = payload.get(name)

        if name in NUMERIC_FIELDS:
            if value is None or (isinstance(value, str) and not value.strip()):
                if spec.get("required"):
                    raise ValueError(f"{name.replace('_', ' ').capitalize()} is required.")
                car[name] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number.")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a valid number.") from None
            if not math.isfinite(number):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a finite number.")

            minimum = spec.get("train_min")
            maximum = spec.get("train_max")
            if minimum is not None and number < minimum:
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be at least {minimum}.")
            if maximum is not None and number > maximum:
                why = " The model was trained only through 2020; later years are outside its training range." if name == "year" else ""
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be no more than {maximum}.{why}")

            if spec.get("dtype") == "int" and not number.is_integer():
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a whole number.")
            if name == "seats" and not number.is_integer():
                raise ValueError("Seats must be a whole number.")
            car[name] = int(number) if spec.get("dtype") == "int" else number
            continue

        if value is None or not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name.replace('_', ' ').capitalize()} is required and must be text.")
        value = value.strip()
        if name != "brand" and "allowed_values" in spec and value not in spec["allowed_values"]:
            allowed = ", ".join(spec["allowed_values"])
            raise ValueError(f"'{value}' is not a valid {name.replace('_', ' ')}. Allowed values: {allowed}.")
        # Unknown brands are intentionally passed to predict.py; its warning explains
        # that the prediction falls back to the model's unknown-brand handling.
        car[name] = value

    return car


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/schema")
def get_schema():
    return jsonify(SCHEMA)


@app.post("/api/predict")
def predict():
    if not request.is_json:
        return _json_error("Send the car details as a JSON request (Content-Type: application/json).")
    payload = request.get_json(silent=True)
    try:
        car = _normalise_and_validate(payload)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        # Import here so the schema endpoint and static frontend remain available even
        # if a local pickle/dependency mismatch prevents model initialisation.
        from predict import predict_with_range

        result = predict_with_range(car)
        return jsonify({
            "price": int(result["price"]),
            "low": int(result["low"]),
            "high": int(result["high"]),
            "warnings": list(result.get("warnings", [])),
        })
    except (ValueError, TypeError, KeyError) as exc:
        return _json_error(f"Could not use these car details: {exc}")
    except Exception:
        app.logger.exception("Prediction failed")
        return _json_error(
            "Prediction could not be completed. Check the installed package versions and "
            "that best_model.pkl is compatible with this Python environment.",
            500,
        )


@app.get("/favicon.ico")
def favicon():
    # Avoid a noisy 404 when browsers automatically request a site icon.
    return "", 204


@app.errorhandler(404)
def not_found(error):
    return _json_error("The requested endpoint was not found.", 404)


@app.errorhandler(500)
def internal_error(error):
    return _json_error("The server encountered an internal error. Check the server console for details.", 500)


if __name__ == "__main__":
    print("Used-car price predictor running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
