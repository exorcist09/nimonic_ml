# Nimonic 263 Machining Surrogate Model

Machine learning pipeline for predicting Nimonic 263 end-milling responses from Abaqus/Explicit simulation or validation data.

The model predicts:

- `RF1` cutting force
- `RF2` thrust force
- `temperature` peak temperature

## Where The Virtual Environment Is Used

Everything is meant to run from this project folder:

```powershell
cd C:\Users\greyy\Downloads\nimonic_ai
```

Create the virtual environment inside the project as `.venv`:

```powershell
py -3.11 -m venv .venv
```

If PowerShell activation is allowed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, use the virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run scripts with:

```powershell
.\.venv\Scripts\python.exe src\train.py
.\.venv\Scripts\python.exe src\predict.py --rake-angle 5 --nose-radius 0.2 --grain-size fine --cutting-speed 0.5
```

So, the work is done in:

- Project folder: `C:\Users\greyy\Downloads\nimonic_ai`
- Virtual environment: `C:\Users\greyy\Downloads\nimonic_ai\.venv`
- Dataset location: `data\raw\machining_data.xlsx`
- Trained model output: `models\random_forest_model.pkl`
- Metrics output: `outputs\metrics\model_metrics.json`
- Plot outputs: `outputs\plots\`

## Dataset Format

Place your Excel file here:

```txt
data/raw/machining_data.xlsx
```

Required columns:

```csv
rake_angle,nose_radius,grain_size,cutting_speed,RF1,RF2,temperature
5,0.2,fine,0.5,2.0,-2.5,330
5,0.4,medium,0.5,1.7,-2.2,312
-5,0.6,coarse,0.5,1.5,-2.0,270
```

## Train Models

```powershell
.\.venv\Scripts\python.exe src\train.py
```

This trains and compares:

- Random Forest Regressor
- Support Vector Regressor
- MLP Regressor

The default production model is Random Forest because it is better suited to small simulation datasets.

## Predict New Conditions

```powershell
.\.venv\Scripts\python.exe src\predict.py --rake-angle 5 --nose-radius 0.4 --grain-size medium --cutting-speed 0.5
```

## Notebook

Start Jupyter from the same virtual environment:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Open:

```txt
notebooks/analysis.ipynb
```

## Outputs

Training generates:

- `data/processed/processed_data.csv`
- `models/random_forest_model.pkl`
- `outputs/metrics/model_metrics.json`
- `outputs/plots/correlation_heatmap.png`
- `outputs/plots/rf1_prediction_vs_actual.png`
- `outputs/plots/rf2_prediction_vs_actual.png`
- `outputs/plots/temperature_prediction_vs_actual.png`
- `outputs/plots/feature_importance.png`
- `outputs/plots/pair_plot.png`
- `outputs/plots/residual_error_plot.png`
