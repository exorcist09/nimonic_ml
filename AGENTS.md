# AGENTS.md

## Project

AI/ML surrogate model for predicting machining responses of Nimonic 263 during end milling.

## Structure

```txt
data/raw/machining_data.xlsx       # User-provided Abaqus/validation dataset
data/processed/processed_data.csv  # Encoded/scaled processed artifact
notebooks/analysis.ipynb           # End-to-end notebook workflow
models/random_forest_model.pkl     # Saved default surrogate model
src/preprocess.py                  # Loading, validation, cleaning, encoding, scaling
src/train.py                       # Model comparison, training, persistence
src/evaluate.py                    # MAE, RMSE, R2 metrics
src/predict.py                     # Reusable CLI prediction script
src/visualize.py                   # Scientific plots for reports
outputs/plots/                     # Generated figures
outputs/metrics/                   # Generated metrics
```

## Environment

Use the project-local virtual environment:

```powershell
cd C:\Users\greyy\Downloads\nimonic_ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prefer `.\.venv\Scripts\python.exe ...` on Windows because PowerShell script execution policy may block activation.

## Dependencies

Required packages are listed in `requirements.txt`:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- notebook
- openpyxl
- joblib

## Coding Conventions

- Keep scripts modular and reusable.
- Use relative paths resolved from the project root.
- Keep preprocessing inside the saved scikit-learn pipeline.
- Preserve required column names: `rake_angle`, `nose_radius`, `grain_size`, `cutting_speed`, `RF1`, `RF2`, `temperature`.
- Use Random Forest as the default saved model unless the user explicitly requests another production model.
- Do not hardcode absolute user paths inside Python modules.

## Execution

Train:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Predict:

```powershell
.\.venv\Scripts\python.exe src\predict.py --rake-angle 5 --nose-radius 0.2 --grain-size fine --cutting-speed 0.5
```

Notebook:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

## Expected Outputs

- Clean processed dataset in `data/processed/`
- Model comparison metrics in `outputs/metrics/`
- Scientific plots in `outputs/plots/`
- Saved Random Forest model in `models/`
- CLI predictions for unseen machining parameters
