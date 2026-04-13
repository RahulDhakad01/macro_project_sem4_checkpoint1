from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "amount",
    "time_gap_minutes",
    "transaction_hour",
    "merchant_risk",
    "distance_from_home_km",
    "device_score",
    "transaction_velocity_24h",
    "previous_declines",
    "account_age_days",
    "is_foreign",
    "is_high_risk_country",
    "card_present",
]

MODEL_VERSION = 4


FEATURE_LABELS = {
    "amount": "Transaction amount",
    "time_gap_minutes": "Minutes since last transaction",
    "transaction_hour": "Hour of transaction",
    "merchant_risk": "Merchant risk score",
    "distance_from_home_km": "Distance from home",
    "device_score": "Device trust score",
    "transaction_velocity_24h": "Transactions in last 24h",
    "previous_declines": "Previous declined transactions",
    "account_age_days": "Account age in days",
    "is_foreign": "Foreign transaction",
    "is_high_risk_country": "High-risk country",
    "card_present": "Card present",
}


RANGES = {
    "amount": {"min": 1, "max": 5000, "step": 1, "default": 1450},
    "time_gap_minutes": {"min": 0, "max": 1440, "step": 1, "default": 4},
    "transaction_hour": {"min": 0, "max": 23, "step": 1, "default": 2},
    "merchant_risk": {"min": 1, "max": 10, "step": 1, "default": 8},
    "distance_from_home_km": {"min": 0, "max": 5000, "step": 1, "default": 920},
    "device_score": {"min": 0, "max": 100, "step": 1, "default": 18},
    "transaction_velocity_24h": {"min": 1, "max": 60, "step": 1, "default": 14},
    "previous_declines": {"min": 0, "max": 10, "step": 1, "default": 3},
    "account_age_days": {"min": 1, "max": 5000, "step": 1, "default": 180},
    "is_foreign": {"min": 0, "max": 1, "step": 1, "default": 1},
    "is_high_risk_country": {"min": 0, "max": 1, "step": 1, "default": 1},
    "card_present": {"min": 0, "max": 1, "step": 1, "default": 0},
}


SCENARIOS = {
    "safe_purchase": {
        "label": "Safe grocery purchase",
        "values": {
            "amount": 62,
            "time_gap_minutes": 420,
            "transaction_hour": 18,
            "merchant_risk": 2,
            "distance_from_home_km": 6,
            "device_score": 88,
            "transaction_velocity_24h": 2,
            "previous_declines": 0,
            "account_age_days": 1460,
            "is_foreign": 0,
            "is_high_risk_country": 0,
            "card_present": 1,
        },
    },
    "travel_booking": {
        "label": "Travel booking",
        "values": {
            "amount": 1850,
            "time_gap_minutes": 190,
            "transaction_hour": 21,
            "merchant_risk": 5,
            "distance_from_home_km": 180,
            "device_score": 76,
            "transaction_velocity_24h": 3,
            "previous_declines": 0,
            "account_age_days": 980,
            "is_foreign": 1,
            "is_high_risk_country": 0,
            "card_present": 0,
        },
    },
    "fraud_attack": {
        "label": "Suspicious midnight attack",
        "values": {
            "amount": 2499,
            "time_gap_minutes": 3,
            "transaction_hour": 1,
            "merchant_risk": 9,
            "distance_from_home_km": 2100,
            "device_score": 12,
            "transaction_velocity_24h": 18,
            "previous_declines": 4,
            "account_age_days": 90,
            "is_foreign": 1,
            "is_high_risk_country": 1,
            "card_present": 0,
        },
    },
}


def generate_synthetic_dataset(rows: int = 6000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    fraud_rows = max(int(rows * 0.09), 1)
    legit_rows = rows - fraud_rows

    legit = pd.DataFrame(
        {
            "amount": np.clip(rng.gamma(shape=2.0, scale=95, size=legit_rows), 1, 2400),
            "time_gap_minutes": np.clip(rng.exponential(scale=260, size=legit_rows), 6, 1440),
            "transaction_hour": rng.choice(
                np.arange(24), size=legit_rows, p=hour_distribution("legitimate")
            ),
            "merchant_risk": rng.integers(1, 7, size=legit_rows),
            "distance_from_home_km": np.clip(
                rng.gamma(shape=1.8, scale=18, size=legit_rows), 0, 600
            ),
            "device_score": np.clip(rng.normal(loc=78, scale=10, size=legit_rows), 35, 100),
            "transaction_velocity_24h": np.clip(
                rng.poisson(lam=2.1, size=legit_rows) + 1, 1, 10
            ),
            "previous_declines": np.clip(rng.poisson(lam=0.12, size=legit_rows), 0, 3),
            "account_age_days": np.clip(
                rng.gamma(shape=4.4, scale=290, size=legit_rows), 120, 5000
            ),
            "is_foreign": rng.binomial(1, 0.08, size=legit_rows),
            "is_high_risk_country": rng.binomial(1, 0.02, size=legit_rows),
            "card_present": rng.binomial(1, 0.68, size=legit_rows),
            "is_fraud": np.zeros(legit_rows, dtype=int),
        }
    )

    fraud = pd.DataFrame(
        {
            "amount": np.clip(rng.gamma(shape=2.4, scale=420, size=fraud_rows), 90, 5000),
            "time_gap_minutes": np.clip(rng.exponential(scale=14, size=fraud_rows), 0, 120),
            "transaction_hour": rng.choice(
                np.arange(24), size=fraud_rows, p=hour_distribution("fraud")
            ),
            "merchant_risk": rng.integers(6, 11, size=fraud_rows),
            "distance_from_home_km": np.clip(
                rng.gamma(shape=2.6, scale=220, size=fraud_rows), 20, 5000
            ),
            "device_score": np.clip(rng.normal(loc=26, scale=12, size=fraud_rows), 0, 70),
            "transaction_velocity_24h": np.clip(
                rng.poisson(lam=8.5, size=fraud_rows) + 1, 3, 60
            ),
            "previous_declines": np.clip(rng.poisson(lam=2.1, size=fraud_rows), 0, 10),
            "account_age_days": np.clip(
                rng.gamma(shape=1.6, scale=120, size=fraud_rows), 1, 1200
            ),
            "is_foreign": rng.binomial(1, 0.66, size=fraud_rows),
            "is_high_risk_country": rng.binomial(1, 0.48, size=fraud_rows),
            "card_present": rng.binomial(1, 0.18, size=fraud_rows),
            "is_fraud": np.ones(fraud_rows, dtype=int),
        }
    )

    frame = pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    noisy_indices = rng.choice(frame.index.to_numpy(), size=max(rows // 30, 1), replace=False)
    frame.loc[noisy_indices, "is_fraud"] = 1 - frame.loc[noisy_indices, "is_fraud"]
    return frame.round(
        {
            "amount": 2,
            "time_gap_minutes": 2,
            "distance_from_home_km": 2,
            "device_score": 2,
        }
    ).reset_index(drop=True)


def hour_distribution(mode: str) -> np.ndarray:
    if mode == "fraud":
        weights = np.array(
            [0.10, 0.11, 0.11, 0.10, 0.08, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
             0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.06, 0.09]
        )
    else:
        weights = np.array(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.06, 0.06, 0.06, 0.06,
             0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.04, 0.03, 0.02]
        )
    return weights / weights.sum()


@dataclass
class ModelArtifacts:
    name: str
    estimator: Any
    metrics: dict[str, float]


class FraudDetectionService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self.model_dir / "fraud_model_bundle.joblib"
        self.report_path = self.model_dir / "training_report.json"
        self.generated_dir = self.model_dir.parent / "static" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.bundle: dict[str, Any] | None = None

    def ensure_ready(self) -> None:
        if self.bundle_path.exists() and self.report_path.exists():
            loaded_bundle = joblib.load(self.bundle_path)
            if loaded_bundle.get("model_version") == MODEL_VERSION:
                self.bundle = loaded_bundle
                return

        self.bundle = self._train_and_store()

    def _train_and_store(self) -> dict[str, Any]:
        dataset = generate_synthetic_dataset()
        X = dataset[FEATURES]
        y = dataset["is_fraud"]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        models = {
            "Logistic Regression + SMOTE": ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=260,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=320,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
        }

        scored_models: list[ModelArtifacts] = []
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        for name, estimator in models.items():
            estimator.fit(X_train, y_train)
            cv_score = float(
                cross_val_score(estimator, X_train_full, y_train_full, cv=cv, scoring="f1").mean()
            )
            val_probabilities = estimator.predict_proba(X_val)[:, 1]
            threshold = best_threshold(y_val.to_numpy(), val_probabilities)
            probabilities = estimator.predict_proba(X_test)[:, 1]
            predicted = (probabilities >= threshold).astype(int)
            scored_models.append(
                ModelArtifacts(
                    name=name,
                    estimator=estimator,
                    metrics={
                        "precision": round(precision_score(y_test, predicted), 3),
                        "recall": round(recall_score(y_test, predicted), 3),
                        "f1": round(f1_score(y_test, predicted), 3),
                        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
                        "avg_precision": round(
                            average_precision_score(y_test, probabilities), 3
                        ),
                        "cv_f1": round(cv_score, 3),
                        "threshold": round(threshold, 3),
                    },
                )
            )

        selected = max(scored_models, key=lambda item: item.metrics["f1"])
        selected_probabilities = selected.estimator.predict_proba(X_test)[:, 1]
        selected_predictions = (selected_probabilities >= selected.metrics["threshold"]).astype(int)
        cm = confusion_matrix(y_test, selected_predictions)
        feature_strength = feature_importance_payload(selected.estimator, X_test, y_test)
        visuals = generate_visuals(self.generated_dir, scored_models, cm, feature_strength)
        bundle = {
            "model_version": MODEL_VERSION,
            "selected_model": selected.name,
            "model": selected.estimator,
            "threshold": selected.metrics["threshold"],
            "metrics": {item.name: item.metrics for item in scored_models},
            "fraud_rate": round(float(dataset["is_fraud"].mean() * 100), 2),
            "dataset_size": int(len(dataset)),
            "feature_labels": FEATURE_LABELS,
            "feature_ranges": RANGES,
            "scenarios": SCENARIOS,
            "feature_importance": feature_strength,
            "visuals": visuals,
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1]),
            },
        }

        joblib.dump(bundle, self.bundle_path)
        self.report_path.write_text(json.dumps(bundle_for_json(bundle), indent=2))
        return bundle

    def model_summary(self) -> dict[str, Any]:
        assert self.bundle is not None
        selected_name = self.bundle["selected_model"]
        return {
            "name": selected_name,
            "metrics": self.bundle["metrics"][selected_name],
        }

    def dashboard_payload(self) -> dict[str, Any]:
        assert self.bundle is not None
        selected = self.model_summary()
        comparison = [
            {
                "name": name,
                "metrics": metrics,
                "selected": name == self.bundle["selected_model"],
            }
            for name, metrics in self.bundle["metrics"].items()
        ]
        return {
            "selected_model": selected,
            "comparison": comparison,
            "fraud_rate": self.bundle["fraud_rate"],
            "dataset_size": self.bundle["dataset_size"],
            "visuals": self.bundle["visuals"],
            "feature_importance": self.bundle["feature_importance"],
            "confusion_matrix": self.bundle["confusion_matrix"],
            "features": [
                {
                    "name": feature,
                    "label": FEATURE_LABELS[feature],
                    **RANGES[feature],
                }
                for feature in FEATURES
            ],
            "scenarios": self.bundle["scenarios"],
        }

    def predict(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        assert self.bundle is not None

        values: dict[str, float] = {}
        for feature in FEATURES:
            if feature not in raw_payload:
                raise ValueError(f"Missing field: {feature}")

            try:
                numeric = float(raw_payload[feature])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for {FEATURE_LABELS[feature]}") from exc

            minimum = RANGES[feature]["min"]
            maximum = RANGES[feature]["max"]
            if numeric < minimum or numeric > maximum:
                raise ValueError(
                    f"{FEATURE_LABELS[feature]} must be between {minimum} and {maximum}"
                )

            if maximum == 1:
                numeric = int(round(numeric))
            values[feature] = numeric

        frame = pd.DataFrame([values], columns=FEATURES)
        model = self.bundle["model"]
        fraud_probability = float(model.predict_proba(frame)[0][1])
        is_fraud = fraud_probability >= self.bundle["threshold"]
        confidence = fraud_probability if is_fraud else 1 - fraud_probability

        top_factors = derive_risk_factors(values, fraud_probability)
        return {
            "prediction": "Fraudulent" if is_fraud else "Legitimate",
            "fraud_probability": round(fraud_probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_band(fraud_probability),
            "top_factors": top_factors,
            "selected_model": self.bundle["selected_model"],
        }


def risk_band(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    if probability >= 0.55:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


def derive_risk_factors(values: dict[str, float], fraud_probability: float) -> list[str]:
    reasons: list[tuple[float, str]] = []

    if values["amount"] > 1500:
        reasons.append((values["amount"] / 1000, "Unusually high transaction amount"))
    if values["time_gap_minutes"] < 10:
        reasons.append((1.8, "Very short time since the previous transaction"))
    if values["transaction_hour"] <= 4 or values["transaction_hour"] >= 23:
        reasons.append((1.5, "Transaction attempted at an unusual hour"))
    if values["merchant_risk"] >= 8:
        reasons.append((1.4, "Merchant category carries elevated fraud risk"))
    if values["distance_from_home_km"] > 400:
        reasons.append((1.3, "Location is far from the customer home area"))
    if values["device_score"] < 35:
        reasons.append((1.7, "Device trust score is unusually low"))
    if values["transaction_velocity_24h"] >= 8:
        reasons.append((1.2, "High transaction count in the last 24 hours"))
    if values["previous_declines"] >= 2:
        reasons.append((1.0, "Recent declines suggest repeated attempts"))
    if values["account_age_days"] < 180:
        reasons.append((0.9, "Account is relatively new"))
    if values["is_foreign"] == 1:
        reasons.append((1.25, "Transaction is marked as foreign"))
    if values["is_high_risk_country"] == 1:
        reasons.append((1.4, "Origin country is tagged as high risk"))
    if values["card_present"] == 0:
        reasons.append((0.7, "Card-not-present transactions are riskier"))

    if not reasons:
        reasons.append((1.0, "Behavior matches the customer normal transaction pattern"))

    reasons.sort(key=lambda item: item[0], reverse=True)
    selected = [reason for _, reason in reasons[:3]]
    if fraud_probability < 0.4 and "Behavior matches the customer normal transaction pattern" not in selected:
        selected = ["Behavior matches the customer normal transaction pattern", *selected[:2]]
    return selected


def bundle_for_json(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": bundle["model_version"],
        "selected_model": bundle["selected_model"],
        "metrics": bundle["metrics"],
        "fraud_rate": bundle["fraud_rate"],
        "dataset_size": bundle["dataset_size"],
        "feature_labels": bundle["feature_labels"],
        "feature_ranges": bundle["feature_ranges"],
        "scenarios": bundle["scenarios"],
        "visuals": bundle["visuals"],
        "feature_importance": bundle["feature_importance"],
        "confusion_matrix": bundle["confusion_matrix"],
    }


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.2, 0.8, 25)
    best_score = -1.0
    best = 0.5
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best_score:
            best_score = score
            best = float(threshold)
    return best


def feature_importance_payload(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float | str]]:
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(FEATURES))

    ranking = sorted(
        [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "importance": round(float(score), 4),
            }
            for feature, score in zip(FEATURES, values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranking[:6]


def bundle_visual_path(name: str) -> str:
    return f"generated/{name}"


def save_plot(fig: plt.Figure, output: Path) -> str:
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return bundle_visual_path(output.name)


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fffaf3",
            "axes.facecolor": "#fffaf3",
            "axes.edgecolor": "#d7c9ba",
            "axes.labelcolor": "#181512",
            "text.color": "#181512",
            "xtick.color": "#6a6259",
            "ytick.color": "#6a6259",
            "font.family": "DejaVu Sans",
        }
    )


def model_metric_rows(scored_models: list[ModelArtifacts]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": item.name,
                "F1-score": item.metrics["f1"],
                "Precision": item.metrics["precision"],
                "Recall": item.metrics["recall"],
                "ROC-AUC": item.metrics["roc_auc"],
            }
            for item in scored_models
        ]
    )


def _color_palette(count: int) -> list[str]:
    base = ["#b85c38", "#1f7a58", "#244c7d", "#8f4021"]
    return base[:count]


def plot_model_comparison(scored_models: list[ModelArtifacts], output_dir: Path) -> str:
    apply_plot_style()
    frame = model_metric_rows(scored_models).set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    frame[["F1-score", "Precision", "Recall", "ROC-AUC"]].plot(
        kind="bar", ax=ax, color=["#b85c38", "#1f7a58", "#244c7d", "#d2a679"]
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=12)
    return save_plot(fig, output_dir / "model_comparison.png")


def plot_confusion_matrix(conf_matrix: np.ndarray, output_dir: Path) -> str:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#b85c38", as_cmap=True),
        cbar=False,
        ax=ax,
    )
    ax.set_title("Selected Model Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=0)
    return save_plot(fig, output_dir / "confusion_matrix.png")


def plot_feature_importance(feature_strength: list[dict[str, float | str]], output_dir: Path) -> str:
    apply_plot_style()
    frame = pd.DataFrame(feature_strength).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(frame["label"], frame["importance"], color=_color_palette(len(frame)))
    ax.set_title("Top Drivers Used By Selected Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return save_plot(fig, output_dir / "feature_importance.png")


def generate_visuals(
    output_dir: Path,
    scored_models: list[ModelArtifacts],
    conf_matrix: np.ndarray,
    feature_strength: list[dict[str, float | str]],
) -> dict[str, str]:
    return {
        "model_comparison": plot_model_comparison(scored_models, output_dir),
        "confusion_matrix": plot_confusion_matrix(conf_matrix, output_dir),
        "feature_importance": plot_feature_importance(feature_strength, output_dir),
    }
