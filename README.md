# Clinical Lab Abnormality Predictor

An end-to-end clinical ML system that predicts diabetes risk from lab values.
Built with production-grade practices: explainability, REST API, experiment tracking,
containerization, drift monitoring, and MIMIC-III integration.

This started as a project to practice clinical ML engineering — the kind of work that
shows up in healthcare data science roles. The Pima Indians dataset is small and
well-understood, which makes it a good sandbox for building the infrastructure
around the model without getting lost in domain complexity.

---

## What's in here

```
clinical-lab-predictor/
├── src/
│   ├── data_curation.py       # Load, clean, impute, QC
│   ├── feature_engineering.py # Clinical features + SMOTE balancing
│   ├── train_rf.py            # Random Forest with GridSearch tuning
│   ├── train_dl.py            # PyTorch feed-forward network
│   ├── evaluate.py            # Metrics + fairness audit across age groups
│   ├── governance.py          # Version registry, audit log, model card
│   ├── mlflow_tracking.py     # Experiment tracking (compare runs in UI)
│   ├── drift_monitor.py       # KS test + PSI for feature/prediction drift
│   ├── dashboard.py           # Streamlit app (predict, explain, monitor)
│   └── mimic_integration.py   # MIMIC-III extraction (+ simulation mode)
├── api/
│   └── main.py                # FastAPI — single + batch predict, SHAP, monitoring
├── tests/
│   └── test_data_curation.py  # Unit tests
├── docs/                      # Generated: model card, eval report, audit log
├── monitoring/                # Generated: prediction logs, drift reports
├── .github/workflows/
│   └── ci_cd.yml              # GitHub Actions: test → train → deploy
├── Dockerfile
└── docker-compose.yml         # API + Dashboard + MLflow + drift monitor
```

---

## Running it

### Option 1: Docker (easiest)

```bash
docker-compose up --build

# API docs:   http://localhost:8000/docs
# Dashboard:  http://localhost:8501
# MLflow UI:  http://localhost:5000
```

### Option 2: Local

```bash
pip install -r requirements.txt

# Run the training pipeline
python src/data_curation.py
python src/feature_engineering.py
python src/train_rf.py
python src/train_dl.py
python src/evaluate.py
python src/governance.py

# Start everything
uvicorn api.main:app --reload --port 8000        # API
streamlit run src/dashboard.py                   # Dashboard
mlflow ui --port 5000                            # MLflow
python src/drift_monitor.py                      # Drift check
```

---

## API

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2, "glucose": 148, "blood_pressure": 72,
    "skin_thickness": 35, "insulin": 50, "bmi": 33.6,
    "diabetes_pedigree": 0.627, "age": 50
  }'

# With SHAP explanation
curl -X POST "http://localhost:8000/predict?explain=true" ...

# Batch
curl -X POST http://localhost:8000/predict/batch \
  -d '{"records": [{...}, {...}]}'
```

---

## Model performance

| Model | AUC-ROC | Accuracy | Sensitivity |
|-------|---------|---------|-------------|
| Random Forest | ~0.84 | ~0.78 | ~0.72 |
| PyTorch DL    | ~0.86 | ~0.80 | ~0.75 |

---

## MIMIC-III

The `mimic_integration.py` module can pull real clinical lab data from a MIMIC-III
PostgreSQL database, or run in simulation mode (no access required):

```python
from src.mimic_integration import run
df = run(use_simulation=True)   # synthetic, works immediately
```

For real data, apply at https://physionet.org/content/mimiciii/1.4/

---

## A note on clinical use

This is a research project, not a clinical tool. The model was trained on Pima Indian
women from Arizona — it almost certainly doesn't generalize to other populations without
retraining and clinical validation. Any real deployment would need prospective validation,
physician sign-off, and regulatory review.

---

**Mayur Patil** — M.S. Data Science, George Washington University  
[LinkedIn](https://linkedin.com/in/mayurpatil26) | [GitHub](https://github.com/mayur212626)
