# Credit Card Fraud Detection System

A demo-ready machine learning project that predicts whether a credit card transaction is fraudulent in real time through a web application.

## What this project includes

- Flask web application with a polished frontend
- Machine learning training pipeline using:
  - Logistic Regression + SMOTE
  - Random Forest
  - Extra Trees
- Synthetic transaction dataset generation for a self-contained demo
- Model comparison using precision, recall, F1-score, ROC-AUC, average precision, and cross-validation
- Real-time fraud prediction API
- Generated evaluation visuals for the presentation
- Sample transaction scenarios for quick classroom demonstration

## Project structure

```text
sem_4/
├── app.py
├── fraud_model.py
├── requirements.txt
├── test_app.py
├── templates/
│   └── index.html
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

## Demo flow suggestion

1. Show the homepage and explain the problem statement.
2. Open the analytics section and explain the selected model, tuned threshold, and cross-validated F1-score.
3. Use the built-in scenarios:
   - Safe grocery purchase
   - Travel booking
   - Suspicious midnight attack
4. Explain the fraud probability, confidence, and top risk factors shown after prediction.
5. Show the generated charts for model comparison, confusion matrix, and feature importance.

## Notes for presentation

- The system is self-contained and does not depend on an external dataset.
- Fraud is treated as an imbalanced classification problem.
- The pipeline uses SMOTE, class balancing, threshold tuning, and model comparison before deployment.
- The app automatically trains and stores the best-performing model on first run.
- The result panel also gives explanation-style risk drivers to make the output easier to justify during viva or presentation.
