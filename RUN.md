# Run the CarValue web app (Windows / Python 3.12)

## 1. Extract and open the project

Extract `car_price_web.zip` to a folder, for example `C:\Users\YourName\Projects\car_price_web`. Open that folder in VS Code. Keep these files together:

- `server.py`, `predict.py`, `car_features.py`
- `best_model.pkl`, `model_input_schema.json`
- `requirements.txt`
- the `static` folder

Do not rename or edit `best_model.pkl`, `car_features.py`, `predict.py`, or `model_input_schema.json`. The saved pickle depends on the feature-engineering class in `car_features.py` and the package versions used when it was created.

## 2. Create a Python 3.12 virtual environment

In the VS Code terminal (PowerShell), run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The original package pins are retained. Flask is the only added direct dependency.

## 3. Start the app

From the project folder, with the virtual environment active:

```powershell
python server.py
```

Open **http://127.0.0.1:5000** in Chrome, Edge, or Firefox. Keep the terminal window open while using the app. No npm, Node.js, or separate frontend build is required.

The server resolves model and schema paths from the location of `server.py`, not from the terminal's current working directory. You can therefore launch it while your terminal is in another folder by using the full path to `server.py`.

## 4. Quick checks

- Select **Load example car**, then **Predict resale value**. The README example is expected to return ₹5,31,480, with a likely range of ₹4,13,492–₹6,63,287.
- Clear one required field and submit to see the inline validation.
- Leave mileage, engine, max power, and seats blank; they are sent as `null` for the model pipeline's median imputation.
- Select **Reset** to clear the form and result.

## Troubleshooting

### `ModuleNotFoundError`

Confirm that the `.venv` is activated in the terminal, then run `python -m pip install -r requirements.txt` again. In VS Code, select the interpreter at `.venv\Scripts\python.exe` using **Python: Select Interpreter**.

### The app cannot find the model or schema / wrong working directory

Keep the model, schema, `predict.py`, `car_features.py`, and `server.py` together in the project root. Start the app with `python server.py`; do not move only the Python script away from the supporting files. The code builds paths relative to `server.py`.

### Port 5000 is already in use

Close the other process using port 5000, then restart the app. Alternatively, change `port=5000` in the final `app.run(...)` line of `server.py` and open the matching port in your browser.

### Pickle / package version mismatch while loading `best_model.pkl`

Recreate the virtual environment with Python 3.12 and install the exact versions in `requirements.txt`. Do not upgrade scikit-learn, XGBoost, NumPy, or pandas independently: the pickle is version-sensitive. If it still fails, capture the full terminal error and compare the installed versions with `pip show numpy pandas scikit-learn xgboost joblib`.

### PowerShell does not allow virtual-environment activation

Use the Command Prompt activation commands in section 2, or configure the PowerShell execution policy according to your institution's Windows policy. Do not run the app outside the intended virtual environment.

### The browser says it cannot connect

Check that `python server.py` is still running and that the address is exactly `http://127.0.0.1:5000`. The page is served by Flask; opening `static/index.html` directly as a file will not work because it needs `/api/schema` and `/api/predict`.
