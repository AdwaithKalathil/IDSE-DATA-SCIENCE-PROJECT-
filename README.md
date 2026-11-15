# Used-Car Resale Price Prediction

An end-to-end machine learning project that predicts the resale price of
used cars in India using listing details from the CarDekho dataset. The
project includes data analysis, model development, an XGBoost prediction
pipeline, and a Flask-based web interface.

**Course:** Introduction to Data Science\
**Tech stack:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP,
Flask, HTML, CSS, JavaScript\
**Random seed:** 42

## 1. Project overview

The application estimates a used car's resale price in Indian rupees
(INR) using details such as:

-   Manufacturing year and kilometres driven
-   Fuel type and transmission
-   Seller type and ownership history
-   Mileage, engine capacity and maximum power
-   Number of seats and vehicle brand

The project also investigates the factors that influence resale value
and how depreciation varies across car brands and price segments.

## 2. Dataset

The project uses the [Vehicle Dataset from
CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho),
published by Nehal Birla on Kaggle. The listings were scraped from the
Indian used-car marketplace CarDekho.

-   Raw dataset: 8,128 rows and 13 columns
-   After cleaning: 6,923 rows
-   Removed during cleaning: 1,202 exact duplicates and 3 impossible
    odometer readings

The dataset is included in the `data/` directory. Refer to the dataset
source for its original details and usage terms.

## 3. Model performance

The following results were obtained on the held-out test set of 693
cars. The models were re-fitted on the combined training and validation
data before final evaluation.

  ------------------------------------------------------------------------------
  Model             RMSE (₹)      MAE (₹)           R²     MAPE (%)        RMSLE
  ------------ ------------- ------------ ------------ ------------ ------------
  Median             529,688      278,974       -0.051         80.8        0.772
  baseline                                                          

  Linear             185,391       97,595        0.871         20.8        0.255
  Regression                                                        

  Random             153,497       76,336        0.912         16.7        0.216
  Forest                                                            
  (tuned)                                                           

  **XGBoost      **156,309**   **74,713**    **0.908**     **16.4**    **0.215**
  (tuned;                                                           
  deployed                                                          
  model)**                                                          
  ------------------------------------------------------------------------------

The XGBoost model was selected based on validation-set performance.
Although Random Forest achieved slightly better RMSE, MAE and R² on the
test set, XGBoost performed better on validation across the reported
metrics and was selected before examining the test results. The two
models were statistically tied in the test-set MAE comparison.

Bootstrap 95% confidence intervals for XGBoost: - MAE:
₹65,000--₹86,000 - MAPE: 15.1--17.8% - R²: 0.85--0.95

### Factors influencing resale value

SHAP and permutation-importance analyses identified car age as the
dominant predictor, followed by maximum power, brand, engine capacity
and fuel type. Once car age is accounted for, kilometres driven, owner
count, transmission and seller type have comparatively smaller effects.

The analysis indicates approximate annual value depreciation of 10% for
budget and mid-range cars, and 12.8% for luxury cars.

## 4. Methodology

### Data cleaning and preprocessing

-   Parsed mileage, engine capacity and maximum power from their
    original string representations.
-   Treated zero mileage and zero maximum power values as missing.
-   Removed the inconsistent torque column.
-   Removed duplicate records before splitting the data.
-   Removed impossible odometer readings.
-   Consolidated the small "Test Drive Car" owner category.
-   Extracted the car brand from the name field.

### Feature engineering

The following features were derived: - `age`: 2026 minus manufacturing
year - `km_per_year`: kilometres driven divided by car age -
`brand_tier`: budget, mid-range or luxury classification based on brand
positioning - `owner_num`: ordinal representation of ownership history

### Model development

The data was divided into training, validation and test sets using an
80:10:10 split, with `random_state=42` and stratification on price
deciles.

Four approaches were evaluated: 1. Median-price baseline 2. Linear
Regression 3. Random Forest 4. XGBoost

Random Forest and XGBoost were tuned using `RandomizedSearchCV` with
five-fold cross-validation on the training set. RMSLE was used as the
tuning score. The target variable was log-transformed using
`TransformedTargetRegressor`; predictions from the saved pipeline are
returned in INR.

Imputation, scaling and one-hot encoding are handled within the
scikit-learn pipeline to avoid data leakage during cross-validation.

## 5. Project structure

The repository contains the analysis notebook, model pipeline,
prediction code and web application. The main files are:

``` text
car_price_project/
├── README.md
├── requirements.txt
├── car_price_prediction.ipynb
├── car_features.py
├── predict.py
├── best_model.pkl
├── model_input_schema.json
├── server.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── Car_details_v3.csv
├── figures/
└── results/
```

-   `car_price_prediction.ipynb`: data exploration, preprocessing, model
    training and evaluation.
-   `car_features.py`: data-cleaning helpers, feature engineering and
    pipeline construction.
-   `predict.py`: loads the saved model and generates predictions from
    raw car details.
-   `best_model.pkl`: fitted preprocessing and XGBoost pipeline.
-   `model_input_schema.json`: input fields, data types, allowed values
    and training ranges.
-   `server.py`: Flask backend serving the web application and
    prediction API.
-   `static/`: HTML, CSS and JavaScript frontend.
-   `figures/` and `results/`: plots and analysis outputs.

## 6. Setup and installation

Python 3.12 is the target environment. Use the dependency versions
specified in `requirements.txt`.

### Windows (PowerShell)

``` powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell activation is unavailable, use Command Prompt:

``` bat
.venv\Scripts\activate.bat
```

### Linux / macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Running the web application

After activating the virtual environment and installing the
dependencies:

``` bash
python server.py
```

Open the following address in a browser:

``` text
http://127.0.0.1:5000
```

Keep the terminal running while using the application. Do not open
`static/index.html` directly; the frontend communicates with the Flask
backend through its API.

## 8. Generating predictions from Python

The prediction pipeline accepts raw input values. Preprocessing,
imputation, feature engineering, encoding and inverse log transformation
are handled by the saved pipeline.

Example:

``` python
from predict import predict_price, predict_with_range

car = {
    "year": 2015,
    "km_driven": 60000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": 23.4,
    "engine": 1248,
    "max_power": 74,
    "seats": 5,
    "brand": "Maruti"
}

price = predict_price(car)
result = predict_with_range(car)

print(price)
print(result)
```

For the example above, the documented output is:

``` text
Prediction: ₹531,480
Range:      ₹413,492 – ₹663,287
Warnings:   []
```

`predict_with_range()` returns a dictionary containing the predicted
price, lower and upper estimates, and any warnings.

## 9. Input fields and constraints

The model expects the following raw fields:

  ---------------------------------------------------------------------------------
  Field             Type              Required          Expected values / range
  ----------------- ----------------- ----------------- ---------------------------
  `year`            Integer           Yes               1983--2020 (training range)

  `km_driven`       Integer           Yes               1,000--577,414

  `fuel`            String            Yes               CNG, Diesel, LPG, Petrol

  `seller_type`     String            Yes               Dealer, Individual,
                                                        Trustmark Dealer

  `transmission`    String            Yes               Automatic, Manual

  `owner`           String            Yes               First Owner, Second Owner,
                                                        Third Owner, Fourth & Above
                                                        Owner

  `brand`           String            Yes               One of the 32 brands in
                                                        `model_input_schema.json`

  `mileage`         Float             No                kmpl (km/kg for CNG/LPG)

  `engine`          Float             No                Engine capacity in CC

  `max_power`       Float             No                Maximum power in bhp

  `seats`           Float             No                2--14
  ---------------------------------------------------------------------------------

The optional numeric fields can be omitted or supplied as `None`;
missing values are imputed by the pipeline. Unseen brands are tolerated
by the prediction pipeline and may generate a warning. Refer to
`model_input_schema.json` for the complete list of supported categorical
values.

## 10. Limitations

1.  The dataset represents historical asking prices from around 2020,
    not current transaction prices. Predictions are in nominal INR.
2.  The training data contains cars with model years only up to 2020.
    Predictions outside the 1983--2020 training range involve
    extrapolation and should be treated cautiously.
3.  The dataset does not include vehicle condition, accident history or
    service records.
4.  Brand representation is uneven: Maruti and Hyundai account for
    approximately 49% of the data, while luxury cars account for around
    2%. Estimates for rare brands and luxury vehicles are therefore less
    reliable.
5.  The model uses brand but not the specific car model or variant.
6.  The test set is relatively small, resulting in uncertainty around
    the reported performance estimates.

## 11. Possible improvements

-   Include car model and variant in the input data.
-   Incorporate vehicle condition, service history and listing date.
-   Explore quantile regression or conformal prediction to calibrate
    prediction intervals.
-   Retrain using a more recent dataset to better reflect current market
    prices.

## 12. Reproducing the analysis

Open `car_price_prediction.ipynb` and run all cells after installing the
dependencies:

``` bash
jupyter notebook car_price_prediction.ipynb
```

Alternatively, execute the notebook from the command line:

``` bash
jupyter nbconvert --to notebook --execute --inplace car_price_prediction.ipynb
```

Running the notebook regenerates the saved model, input schema, figures
and result tables. The documented results use `random_state=42` and the
dependency versions pinned in `requirements.txt`.
