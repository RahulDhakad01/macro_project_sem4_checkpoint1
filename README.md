# Credit Card Fraud Detection System

A hybrid fraud detection platform that combines Indian transaction behavior modeling with European card-fraud anomaly detection inside a single web application.

## What this project includes

- Flask web application with a responsive frontend for desktop, tablet, and phone
- Hybrid machine learning pipeline using:
  - Indian behavioral fraud model
  - Global card-pattern fraud model
  - Fusion scoring layer
- Real dataset training using:
  - `Updated_Inclusive_Indian_Online_Scam_Dataset (1).csv`
  - `creditcard.csv` from the European Card Fraud Dataset
- Real-time fraud prediction API with dynamic reason generation
- Separate metrics page for architecture, charts, confusion matrix, feature drivers, and methodology
- Render deployment blueprint via `render.yaml`

## Project structure

```text
sem_4/
├── app.py
├── fraud_model.py
├── requirements.txt
├── render.yaml
├── test_app.py
├── templates/
│   ├── index.html
│   └── metrics.html
└── static/
    ├── app.js
    └── styles.css
```

## How to run

1. Create and activate a virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the application:

```bash
python3 app.py
```

4. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Deployment on Render

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repo.
3. Render will read `render.yaml`, install dependencies, and start the app with Gunicorn.
4. Use `/api/health` as the health endpoint and `/metrics` for the model insights page.

## Key pages

- `/` — unified fraud detector
- `/metrics` — model metrics, evidence, confusion matrix, feature importance, and methodology
- `/api/health` — health check
- `/api/predict` — fraud prediction API

## Notes for presentation

- The deployed app runs from prebuilt model artifacts, so production does not depend on your local CSV paths.
- The Indian and European models are trained independently and combined through a weighted fusion layer.
- The prediction response returns a single fraud score, risk level, and dynamic explanation list.
- The interface is responsive and optimized for smaller screens as well as desktop layouts.
