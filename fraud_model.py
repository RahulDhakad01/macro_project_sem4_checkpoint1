from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict, deque
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_VERSION = 5
INDIAN_DATASET_PATH = Path(
    "/Users/rahuldhakad/Downloads/Updated_Inclusive_Indian_Online_Scam_Dataset (1).csv"
)
EUROPEAN_DATASET_PATH = Path("/Users/rahuldhakad/Downloads/creditcard.csv")
FUSION_WEIGHTS = {"indian": 0.6, "global": 0.4}

UNIFIED_FIELDS = [
    "transaction_amount",
    "transaction_hour",
    "location",
    "merchant_category",
    "card_type",
    "transactions_last_24h",
    "previous_declined_transactions",
    "distance_from_home",
    "foreign_transaction",
    "card_present",
]

FIELD_LABELS = {
    "transaction_amount": "Transaction amount",
    "transaction_hour": "Transaction hour",
    "location": "Location",
    "merchant_category": "Merchant category",
    "card_type": "Card type",
    "transactions_last_24h": "Transactions in last 24h",
    "previous_declined_transactions": "Previous declined transactions",
    "distance_from_home": "Distance from home (km)",
    "foreign_transaction": "Foreign transaction",
    "card_present": "Card present",
}

INDIAN_MODEL_FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "location",
    "merchant_category",
    "card_type",
    "transactions_last_24h",
    "previous_declined_transactions",
    "distance_from_home",
    "foreign_transaction",
    "card_present",
    "is_night",
    "high_amount",
    "txn_velocity",
    "location_change",
    "device_trust_score",
]

GLOBAL_MODEL_FEATURES = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]

CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Surat": (21.1702, 72.8311),
}

UNIFIED_FIELD_CONFIG = {
    "transaction_amount": {
        "input_type": "number",
        "min": 1,
        "max": 50000,
        "step": 0.01,
        "default": 3200,
    },
    "transaction_hour": {
        "input_type": "number",
        "min": 0,
        "max": 23,
        "step": 1,
        "default": 1,
    },
    "location": {"input_type": "select"},
    "merchant_category": {"input_type": "select"},
    "card_type": {"input_type": "select"},
    "transactions_last_24h": {
        "input_type": "number",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 9,
    },
    "previous_declined_transactions": {
        "input_type": "number",
        "min": 0,
        "max": 10,
        "step": 1,
        "default": 2,
    },
    "distance_from_home": {
        "input_type": "number",
        "min": 0,
        "max": 10000,
        "step": 1,
        "default": 780,
    },
    "foreign_transaction": {"input_type": "select"},
    "card_present": {"input_type": "select"},
}

SCENARIOS = {
    "safe_purchase": {
        "label": "Safe local purchase",
        "values": {
            "transaction_amount": 2400,
            "transaction_hour": 15,
            "location": "Pune",
            "merchant_category": "POS",
            "card_type": "Visa",
            "transactions_last_24h": 2,
            "previous_declined_transactions": 0,
            "distance_from_home": 6,
            "foreign_transaction": "0",
            "card_present": "1",
        },
    },
    "digital_anomaly": {
        "label": "Digital anomaly",
        "values": {
            "transaction_amount": 8600,
            "transaction_hour": 2,
            "location": "Delhi",
            "merchant_category": "Digital",
            "card_type": "MasterCard",
            "transactions_last_24h": 11,
            "previous_declined_transactions": 3,
            "distance_from_home": 940,
            "foreign_transaction": "1",
            "card_present": "0",
        },
    },
    "travel_spike": {
        "label": "Travel spike",
        "values": {
            "transaction_amount": 12500,
            "transaction_hour": 23,
            "location": "Bangalore",
            "merchant_category": "Digital",
            "card_type": "Rupay",
            "transactions_last_24h": 14,
            "previous_declined_transactions": 2,
            "distance_from_home": 1450,
            "foreign_transaction": "1",
            "card_present": "0",
        },
    },
}


@dataclass
class ModelArtifacts:
    name: str
    estimator: Any
    metrics: dict[str, float]
    threshold: float


class FraudDetectionService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self.model_dir / "fraud_model_bundle.joblib"
        self.report_path = self.model_dir / "training_report.json"
        self.generated_dir = self.model_dir.parent / "static" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(self.model_dir / "prediction.log")
        self.bundle: dict[str, Any] | None = None

    def ensure_ready(self) -> None:
        if self.bundle_path.exists() and self.report_path.exists():
            loaded_bundle = joblib.load(self.bundle_path)
            current_meta = self._dataset_meta()
            if loaded_bundle.get("model_version") == MODEL_VERSION and (
                current_meta is None or loaded_bundle.get("dataset_meta") == current_meta
            ):
                self.bundle = loaded_bundle
                return

        self.bundle = self._train_and_store()

    def _dataset_meta(self) -> dict[str, float | int | str] | None:
        if not (INDIAN_DATASET_PATH.exists() and EUROPEAN_DATASET_PATH.exists()):
            return None
        return {
            "indian_path": str(INDIAN_DATASET_PATH),
            "indian_size": INDIAN_DATASET_PATH.stat().st_size,
            "indian_mtime": int(INDIAN_DATASET_PATH.stat().st_mtime),
            "european_path": str(EUROPEAN_DATASET_PATH),
            "european_size": EUROPEAN_DATASET_PATH.stat().st_size,
            "european_mtime": int(EUROPEAN_DATASET_PATH.stat().st_mtime),
        }

    def _train_and_store(self) -> dict[str, Any]:
        ensure_dataset_exists(INDIAN_DATASET_PATH)
        ensure_dataset_exists(EUROPEAN_DATASET_PATH)

        indian_raw = pd.read_csv(INDIAN_DATASET_PATH)
        indian_frame, indian_reference = prepare_indian_dataset(indian_raw)
        indian_model, indian_confusion, feature_importance = train_indian_model(indian_frame)

        european_raw = pd.read_csv(EUROPEAN_DATASET_PATH)
        european_frame, global_reference = prepare_european_dataset(european_raw)
        global_model = train_global_model(european_frame)

        visuals = generate_visuals(
            self.generated_dir,
            [indian_model, global_model],
            indian_confusion,
            feature_importance,
        )

        bundle = {
            "model_version": MODEL_VERSION,
            "system_name": "Hybrid Fusion Engine",
            "dataset_meta": self._dataset_meta(),
            "fusion_weights": FUSION_WEIGHTS,
            "indian_model": indian_model.estimator,
            "indian_threshold": indian_model.threshold,
            "global_model": global_model.estimator,
            "global_threshold": global_model.threshold,
            "metrics": {
                "Indian Behavioral Model": indian_model.metrics,
                "Global Card Pattern Model": global_model.metrics,
            },
            "dataset_summary": {
                "indian_rows": int(len(indian_frame)),
                "indian_fraud_rate": round(float(indian_frame["target"].mean() * 100), 2),
                "european_rows": int(len(european_frame)),
                "european_fraud_rate": round(float(european_frame["Class"].mean() * 100), 3),
            },
            "indian_reference": indian_reference,
            "global_reference": global_reference,
            "feature_importance": feature_importance,
            "visuals": visuals,
            "confusion_matrix": {
                "true_negative": int(indian_confusion[0, 0]),
                "false_positive": int(indian_confusion[0, 1]),
                "false_negative": int(indian_confusion[1, 0]),
                "true_positive": int(indian_confusion[1, 1]),
            },
            "form_options": build_form_options(indian_reference),
            "scenarios": SCENARIOS,
        }

        joblib.dump(bundle, self.bundle_path)
        self.report_path.write_text(json.dumps(bundle_for_json(bundle), indent=2))
        return bundle

    def model_summary(self) -> dict[str, Any]:
        assert self.bundle is not None
        indian = self.bundle["metrics"]["Indian Behavioral Model"]
        global_metrics = self.bundle["metrics"]["Global Card Pattern Model"]
        hybrid_quality = round((indian["f1"] + global_metrics["f1"]) / 2, 3)
        return {
            "name": self.bundle["system_name"],
            "metrics": {"f1": hybrid_quality},
        }

    def dashboard_payload(self) -> dict[str, Any]:
        assert self.bundle is not None

        comparison = [
            {
                "name": "Indian Behavioral Model",
                "metrics": self.bundle["metrics"]["Indian Behavioral Model"],
                "selected": False,
            },
            {
                "name": "Global Card Pattern Model",
                "metrics": self.bundle["metrics"]["Global Card Pattern Model"],
                "selected": False,
            },
        ]

        fusion = self.bundle["fusion_weights"]
        datasets = self.bundle["dataset_summary"]

        return {
            "system_name": self.bundle["system_name"],
            "selected_model": {
                "name": self.bundle["system_name"],
                "metrics": {
                    "f1": self.bundle["metrics"]["Indian Behavioral Model"]["f1"],
                    "precision": self.bundle["metrics"]["Indian Behavioral Model"]["precision"],
                    "recall": self.bundle["metrics"]["Indian Behavioral Model"]["recall"],
                    "cv_f1": self.bundle["metrics"]["Indian Behavioral Model"]["cv_f1"],
                    "avg_precision": self.bundle["metrics"]["Global Card Pattern Model"][
                        "avg_precision"
                    ],
                    "threshold": round(
                        fusion["indian"] * self.bundle["indian_threshold"]
                        + fusion["global"] * self.bundle["global_threshold"],
                        3,
                    ),
                },
            },
            "comparison": comparison,
            "datasets": datasets,
            "fusion_weights": fusion,
            "visuals": self.bundle["visuals"],
            "feature_importance": self.bundle["feature_importance"],
            "confusion_matrix": self.bundle["confusion_matrix"],
            "features": build_dashboard_features(self.bundle["form_options"]),
            "scenarios": self.bundle["scenarios"],
        }

    def predict(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        assert self.bundle is not None

        validated = validate_unified_payload(raw_payload, self.bundle["form_options"])
        engineered = engineer_unified_features(validated, self.bundle["indian_reference"])

        indian_frame = pd.DataFrame(
            [{feature: engineered[feature] for feature in INDIAN_MODEL_FEATURES}],
            columns=INDIAN_MODEL_FEATURES,
        )
        p_indian = float(self.bundle["indian_model"].predict_proba(indian_frame)[0][1])

        global_input, global_risk_proxy = build_global_input(
            validated,
            engineered,
            self.bundle["global_reference"],
            self.bundle["indian_reference"],
        )
        global_frame = pd.DataFrame([global_input], columns=GLOBAL_MODEL_FEATURES)
        raw_global_probability = float(self.bundle["global_model"].predict_proba(global_frame)[0][1])
        p_global = calibrate_global_probability(raw_global_probability, global_risk_proxy)

        final_score = (
            self.bundle["fusion_weights"]["indian"] * p_indian
            + self.bundle["fusion_weights"]["global"] * p_global
        )
        risk_level = fusion_risk_band(final_score)
        confidence = max(final_score, 1 - final_score)

        reasons = derive_dynamic_reasons(
            validated,
            engineered,
            p_indian,
            p_global,
            final_score,
            self.bundle["indian_reference"],
        )

        note = None
        if p_indian > p_global and engineered["location_change"] == 1:
            note = "Behavior deviates from the typical Indian transaction pattern."

        self.logger.info(
            json.dumps(
                {
                    "input": validated,
                    "engineered": {
                        "is_night": engineered["is_night"],
                        "high_amount": engineered["high_amount"],
                        "location_change": engineered["location_change"],
                        "device_trust_score": engineered["device_trust_score"],
                    },
                    "P_indian": round(p_indian, 4),
                    "P_global": round(p_global, 4),
                    "P_global_raw": round(raw_global_probability, 4),
                    "global_risk_proxy": round(global_risk_proxy, 4),
                    "fraud_score": round(final_score, 4),
                    "risk_level": risk_level,
                }
            )
        )

        return {
            "prediction": f"{risk_level} Risk",
            "fraud_score": round(final_score * 100, 2),
            "fraud_probability": round(final_score * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "P_indian": round(p_indian, 4),
            "P_global": round(p_global, 4),
            "reasons": reasons,
            "top_factors": reasons,
            "selected_model": self.bundle["system_name"],
            "message": note
            or "Fusion engine combined Indian behavior and global card patterns for this decision.",
        }


def ensure_dataset_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("fraud_prediction_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        if os.access(log_path.parent, os.W_OK):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    except OSError:
        pass
    return logger


def prepare_indian_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw.copy()
    frame = frame.dropna(subset=["is_fraudulent"]).copy()
    frame["is_fraudulent"] = frame["is_fraudulent"].astype(int)
    frame["transaction_time"] = pd.to_datetime(
        frame["transaction_time"], errors="coerce", format="%m/%d/%Y %H:%M"
    )
    frame = frame.dropna(subset=["transaction_time"]).copy()

    frame["location"] = frame["location"].fillna("Unknown").astype(str)
    frame["merchant_category"] = frame["purchase_category"].fillna("Unknown").astype(str)
    frame["card_type"] = frame["card_type"].fillna("Unknown").astype(str)
    frame["transaction_amount"] = frame["amount"].astype(float)
    frame["transaction_hour"] = frame["transaction_time"].dt.hour.astype(int)

    home_locations = (
        frame.groupby("customer_id")["location"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else "Unknown")
        .to_dict()
    )
    frame["home_location"] = frame["customer_id"].map(home_locations).fillna(frame["location"])
    frame["distance_from_home"] = frame.apply(
        lambda row: city_distance_km(row["location"], row["home_location"]), axis=1
    )

    history = historical_activity_features(frame)
    frame["transactions_last_24h"] = history["transactions_last_24h"]
    frame["previous_declined_transactions"] = history["previous_declined_transactions"]

    frame["foreign_transaction"] = frame["location"].apply(
        lambda city: 0 if city in CITY_COORDS else 1
    )
    frame["card_present"] = frame["merchant_category"].apply(lambda value: 1 if value == "POS" else 0)

    amount_threshold = float(frame["transaction_amount"].quantile(0.9))
    frame["is_night"] = (frame["transaction_hour"] < 6).astype(int)
    frame["high_amount"] = (frame["transaction_amount"] > amount_threshold).astype(int)
    frame["txn_velocity"] = frame["transactions_last_24h"].astype(float)
    frame["location_change"] = (frame["distance_from_home"] > 50).astype(int)
    frame["device_trust_score"] = frame.apply(
        lambda row: device_trust_score(
            distance_from_home=float(row["distance_from_home"]),
            transactions_last_24h=float(row["transactions_last_24h"]),
            previous_declined_transactions=float(row["previous_declined_transactions"]),
            foreign_transaction=int(row["foreign_transaction"]),
            card_present=int(row["card_present"]),
            location_change=int(row["location_change"]),
        ),
        axis=1,
    )

    location_risk = frame.groupby("location")["is_fraudulent"].mean().to_dict()
    merchant_risk = frame.groupby("merchant_category")["is_fraudulent"].mean().to_dict()
    card_type_risk = frame.groupby("card_type")["is_fraudulent"].mean().to_dict()

    dataset = frame[INDIAN_MODEL_FEATURES].copy()
    dataset["target"] = frame["is_fraudulent"].astype(int)

    reference = {
        "amount_threshold": round(amount_threshold, 2),
        "location_options": sorted(frame["location"].dropna().unique().tolist()),
        "merchant_options": sorted(frame["merchant_category"].dropna().unique().tolist()),
        "card_type_options": sorted(frame["card_type"].dropna().unique().tolist()),
        "location_risk": location_risk,
        "merchant_risk": merchant_risk,
        "card_type_risk": card_type_risk,
        "overall_fraud_rate": float(frame["is_fraudulent"].mean()),
    }
    return dataset, reference


def historical_activity_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("transaction_time").copy()
    tx_windows: dict[Any, deque[pd.Timestamp]] = defaultdict(deque)
    previous_declines: dict[Any, int] = defaultdict(int)

    counts: list[int] = []
    declines: list[int] = []

    for _, row in ordered.iterrows():
        customer_id = row["customer_id"]
        current_time = row["transaction_time"]
        window = tx_windows[customer_id]
        while window and (current_time - window[0]).total_seconds() > 86400:
            window.popleft()

        counts.append(len(window))
        declines.append(previous_declines[customer_id])

        window.append(current_time)
        previous_declines[customer_id] += int(row["is_fraudulent"])

    ordered["transactions_last_24h"] = counts
    ordered["previous_declined_transactions"] = np.clip(declines, 0, 10)
    return ordered.sort_index()[["transactions_last_24h", "previous_declined_transactions"]]


def city_distance_km(city_a: str, city_b: str) -> float:
    if city_a == city_b:
        return 0.0
    if city_a not in CITY_COORDS or city_b not in CITY_COORDS:
        return 350.0
    lat1, lon1 = CITY_COORDS[city_a]
    lat2, lon2 = CITY_COORDS[city_b]
    return round(haversine_km(lat1, lon1, lat2, lon2), 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def prepare_european_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    fraud = raw[raw["Class"] == 1]
    non_fraud = raw[raw["Class"] == 0].sample(n=25000, random_state=42)
    sampled = pd.concat([fraud, non_fraud], ignore_index=True).sample(frac=1, random_state=42)

    v_columns = [f"V{i}" for i in range(1, 29)]
    reference = {
        "amount_threshold": round(float(raw["Amount"].quantile(0.9)), 2),
        "amount_q99": round(float(raw["Amount"].quantile(0.99)), 2),
        "safe_amount_median": round(float(raw[raw["Class"] == 0]["Amount"].median()), 2),
        "fraud_amount_median": round(float(raw[raw["Class"] == 1]["Amount"].median()), 2),
        "time_max": float(raw["Time"].max()),
        "fraud_medians": raw[raw["Class"] == 1][v_columns].median().to_dict(),
        "safe_medians": raw[raw["Class"] == 0][v_columns].median().to_dict(),
        "v_stds": raw[v_columns].std().replace(0, 1).to_dict(),
    }
    return sampled[GLOBAL_MODEL_FEATURES + ["Class"]].copy(), reference


def train_indian_model(
    dataset: pd.DataFrame,
) -> tuple[ModelArtifacts, np.ndarray, list[dict[str, float | str]]]:
    X = dataset[INDIAN_MODEL_FEATURES]
    y = dataset["target"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    categorical = ["location", "merchant_category", "card_type"]
    numeric = [feature for feature in INDIAN_MODEL_FEATURES if feature not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    estimator = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=320,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    estimator.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_score = float(cross_val_score(estimator, X_train_full, y_train_full, cv=cv, scoring="f1").mean())

    val_probabilities = estimator.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val.to_numpy(), val_probabilities)

    probabilities = estimator.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "precision": round(precision_score(y_test, predicted, zero_division=0), 3),
        "recall": round(recall_score(y_test, predicted, zero_division=0), 3),
        "f1": round(f1_score(y_test, predicted, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
        "avg_precision": round(average_precision_score(y_test, probabilities), 3),
        "cv_f1": round(cv_score, 3),
        "threshold": round(threshold, 3),
    }

    confusion = confusion_matrix(y_test, predicted)
    importance = aggregate_feature_importance(estimator, categorical, numeric)
    return (
        ModelArtifacts(
            name="Indian Behavioral Model",
            estimator=estimator,
            metrics=metrics,
            threshold=threshold,
        ),
        confusion,
        importance,
    )


def train_global_model(dataset: pd.DataFrame) -> ModelArtifacts:
    X = dataset[GLOBAL_MODEL_FEATURES]
    y = dataset["Class"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    estimator.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_score = float(cross_val_score(estimator, X_train_full, y_train_full, cv=cv, scoring="f1").mean())

    val_probabilities = estimator.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val.to_numpy(), val_probabilities)

    probabilities = estimator.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "precision": round(precision_score(y_test, predicted, zero_division=0), 3),
        "recall": round(recall_score(y_test, predicted, zero_division=0), 3),
        "f1": round(f1_score(y_test, predicted, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
        "avg_precision": round(average_precision_score(y_test, probabilities), 3),
        "cv_f1": round(cv_score, 3),
        "threshold": round(threshold, 3),
    }
    return ModelArtifacts(
        name="Global Card Pattern Model",
        estimator=estimator,
        metrics=metrics,
        threshold=threshold,
    )


def aggregate_feature_importance(
    estimator: Pipeline, categorical: list[str], numeric: list[str]
) -> list[dict[str, float | str]]:
    preprocessor: ColumnTransformer = estimator.named_steps["preprocessor"]
    model: RandomForestClassifier = estimator.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    raw_scores = model.feature_importances_

    grouped: defaultdict[str, float] = defaultdict(float)
    for feature_name, score in zip(feature_names, raw_scores):
        clean = feature_name.split("__", 1)[1]
        raw_name = clean.split("_", 1)[0] if clean.split("_", 1)[0] in categorical else clean
        grouped[raw_name] += float(score)

    ordered = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "feature": name,
            "label": FIELD_LABELS.get(name, name.replace("_", " ").title()),
            "importance": round(score, 4),
        }
        for name, score in ordered[:6]
    ]


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.2, 0.8, 25)
    best_score = -1.0
    chosen = 0.5
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best_score:
            best_score = score
            chosen = float(threshold)
    return chosen


def build_dashboard_features(form_options: dict[str, list[str]]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for field in UNIFIED_FIELDS:
        config = dict(UNIFIED_FIELD_CONFIG[field])
        payload = {"name": field, "label": FIELD_LABELS[field], **config}
        if config["input_type"] == "select":
            if field in {"foreign_transaction", "card_present"}:
                payload["options"] = [
                    {"value": "0", "label": "0 - No"},
                    {"value": "1", "label": "1 - Yes"},
                ]
                payload["default"] = "0"
            else:
                payload["options"] = [
                    {"value": option, "label": option} for option in form_options[field]
                ]
                payload["default"] = form_options[field][0]
        features.append(payload)
    return features


def build_form_options(indian_reference: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "location": indian_reference["location_options"],
        "merchant_category": indian_reference["merchant_options"],
        "card_type": indian_reference["card_type_options"],
    }


def validate_unified_payload(
    raw_payload: dict[str, Any], form_options: dict[str, list[str]]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    numeric_fields = {
        "transaction_amount": (1, 50000),
        "transaction_hour": (0, 23),
        "transactions_last_24h": (0, 50),
        "previous_declined_transactions": (0, 10),
        "distance_from_home": (0, 10000),
    }

    for field in UNIFIED_FIELDS:
        if field not in raw_payload:
            raise ValueError(f"Missing field: {field}")

    for field, (minimum, maximum) in numeric_fields.items():
        try:
            numeric = float(raw_payload[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {FIELD_LABELS[field]}") from exc
        if numeric < minimum or numeric > maximum:
            raise ValueError(
                f"{FIELD_LABELS[field]} must be between {minimum} and {maximum}"
            )
        values[field] = int(numeric) if field != "transaction_amount" else float(numeric)

    for field in ["location", "merchant_category", "card_type"]:
        candidate = str(raw_payload[field])
        if candidate not in form_options[field]:
            raise ValueError(f"Invalid option for {FIELD_LABELS[field]}")
        values[field] = candidate

    for field in ["foreign_transaction", "card_present"]:
        candidate = str(raw_payload[field])
        if candidate not in {"0", "1", 0, 1}:
            raise ValueError(f"{FIELD_LABELS[field]} must be 0 or 1")
        values[field] = int(candidate)

    return values


def engineer_unified_features(
    values: dict[str, Any], indian_reference: dict[str, Any]
) -> dict[str, Any]:
    engineered = dict(values)
    engineered["is_night"] = int(values["transaction_hour"] < 6)
    engineered["high_amount"] = int(
        values["transaction_amount"] > indian_reference["amount_threshold"]
    )
    engineered["txn_velocity"] = values["transactions_last_24h"]
    engineered["location_change"] = int(
        values["distance_from_home"] > 50 or values["foreign_transaction"] == 1
    )
    engineered["device_trust_score"] = device_trust_score(
        distance_from_home=values["distance_from_home"],
        transactions_last_24h=values["transactions_last_24h"],
        previous_declined_transactions=values["previous_declined_transactions"],
        foreign_transaction=values["foreign_transaction"],
        card_present=values["card_present"],
        location_change=engineered["location_change"],
    )
    return engineered


def device_trust_score(
    *,
    distance_from_home: float,
    transactions_last_24h: float,
    previous_declined_transactions: float,
    foreign_transaction: int,
    card_present: int,
    location_change: int,
) -> float:
    score = (
        88
        - min(distance_from_home / 18, 35)
        - min(transactions_last_24h * 2.6, 20)
        - min(previous_declined_transactions * 6, 18)
        - 14 * foreign_transaction
        - 8 * location_change
        + 7 * card_present
    )
    return round(float(np.clip(score, 5, 98)), 2)


def build_global_input(
    values: dict[str, Any],
    engineered: dict[str, Any],
    global_reference: dict[str, Any],
    indian_reference: dict[str, Any],
) -> tuple[dict[str, float], float]:
    risk_proxy = global_structural_risk(values, engineered)

    amount = mapped_global_amount(values, engineered, global_reference, indian_reference, risk_proxy)
    time_seconds = float(
        min(global_reference["time_max"], values["transaction_hour"] * 3600 + values["transactions_last_24h"] * 45)
    )
    seed = (
        values["transaction_hour"] * 0.7
        + amount * 0.004
        + values["transactions_last_24h"] * 0.9
        + values["previous_declined_transactions"] * 1.3
        + values["distance_from_home"] * 0.0015
        + values["foreign_transaction"] * 1.7
        + (1 - values["card_present"]) * 0.8
    )

    mapped: dict[str, float] = {"Time": time_seconds, "Amount": amount}
    for index in range(1, 29):
        key = f"V{index}"
        safe = global_reference["safe_medians"][key]
        fraud = global_reference["fraud_medians"][key]
        std = global_reference["v_stds"][key]
        oscillation = math.sin(seed * (index + 1))
        curve = risk_proxy ** 1.8
        mapped[key] = safe + curve * (fraud - safe) + oscillation * std * (0.01 + 0.015 * risk_proxy)
    return mapped, risk_proxy


def global_structural_risk(values: dict[str, Any], engineered: dict[str, Any]) -> float:
    return float(
        np.clip(
            0.24 * engineered["high_amount"]
            + 0.11 * engineered["is_night"]
            + 0.16 * values["foreign_transaction"]
            + 0.10 * (1 - values["card_present"])
            + 0.10 * min(values["transactions_last_24h"] / 16, 1)
            + 0.08 * min(values["previous_declined_transactions"] / 5, 1)
            + 0.07 * min(values["distance_from_home"] / 2500, 1)
            + 0.14 * max(0, (45 - engineered["device_trust_score"]) / 45),
            0,
            1,
        )
    )


def mapped_global_amount(
    values: dict[str, Any],
    engineered: dict[str, Any],
    global_reference: dict[str, Any],
    indian_reference: dict[str, Any],
    risk_proxy: float,
) -> float:
    safe_amount = float(global_reference.get("safe_amount_median", 22.0))
    amount_ceiling = float(
        max(
            global_reference.get("amount_q99", global_reference["amount_threshold"] * 5),
            global_reference["amount_threshold"] + 1,
        )
    )
    baseline_threshold = max(float(indian_reference["amount_threshold"]), 1.0)
    amount_ratio = min(values["transaction_amount"] / baseline_threshold, 3.0) / 3.0
    if engineered["high_amount"] == 0:
        amount_ratio *= 0.35
    amount_blend = np.clip(0.7 * risk_proxy + 0.3 * amount_ratio, 0, 1)
    return float(safe_amount + (amount_blend ** 1.6) * (amount_ceiling - safe_amount))


def calibrate_global_probability(raw_probability: float, risk_proxy: float) -> float:
    blend = 0.18 + 0.72 * risk_proxy
    calibrated = blend * raw_probability + (1 - blend) * risk_proxy
    if risk_proxy < 0.12:
        calibrated = min(calibrated, 0.18)
    elif risk_proxy < 0.2:
        calibrated = min(calibrated, 0.28)
    return float(np.clip(calibrated, 0.01, 0.99))


def fusion_risk_band(score: float) -> str:
    if score > 0.7:
        return "HIGH"
    if score >= 0.3:
        return "MEDIUM"
    return "LOW"


def derive_dynamic_reasons(
    values: dict[str, Any],
    engineered: dict[str, Any],
    p_indian: float,
    p_global: float,
    final_score: float,
    indian_reference: dict[str, Any],
) -> list[str]:
    reasons: list[tuple[float, str]] = []

    if engineered["is_night"] == 1:
        reasons.append((1.0, "Transaction at an unusual hour"))
    if engineered["high_amount"] == 1:
        reasons.append((1.2, "High transaction amount compared with typical behavior"))
    if engineered["location_change"] == 1:
        reasons.append((1.15, "New location detected relative to the home area"))
    if values["foreign_transaction"] == 1:
        reasons.append((1.1, "Foreign transaction flag is active"))
    if values["transactions_last_24h"] >= 8:
        reasons.append((1.0, "High transaction frequency in the last 24 hours"))
    if values["previous_declined_transactions"] >= 2:
        reasons.append((0.95, "Multiple previous declined attempts were reported"))
    if engineered["device_trust_score"] <= 40:
        reasons.append((1.05, "Low device trust score based on transaction behavior"))
    if values["card_present"] == 0:
        reasons.append((0.7, "Card-not-present activity increases digital fraud risk"))

    merchant_risk = indian_reference["merchant_risk"].get(values["merchant_category"], 0)
    if merchant_risk > indian_reference["overall_fraud_rate"] + 0.08:
        reasons.append((0.85, "Merchant category shows elevated fraud risk in Indian data"))

    location_risk = indian_reference["location_risk"].get(values["location"], 0)
    if location_risk > indian_reference["overall_fraud_rate"] + 0.07:
        reasons.append((0.8, "Current location has higher fraud incidence in Indian data"))

    if p_indian > p_global and final_score >= 0.45:
        reasons.append((0.9, "Behavior deviates from typical Indian transaction patterns"))
    elif p_global > 0.55:
        reasons.append((0.75, "Global card fraud model detected abnormal card usage patterns"))

    if not reasons:
        reasons.append((1.0, "Behavior remains close to the normal low-risk transaction profile"))

    reasons.sort(key=lambda item: item[0], reverse=True)
    return [reason for _, reason in reasons[:4]]


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


def save_plot(fig: plt.Figure, output: Path) -> str:
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f"generated/{output.name}"


def generate_visuals(
    output_dir: Path,
    models: list[ModelArtifacts],
    indian_confusion: np.ndarray,
    feature_importance: list[dict[str, float | str]],
) -> dict[str, str]:
    return {
        "model_comparison": plot_model_comparison(models, output_dir),
        "confusion_matrix": plot_confusion_matrix(indian_confusion, output_dir),
        "feature_importance": plot_feature_importance(feature_importance, output_dir),
    }


def plot_model_comparison(models: list[ModelArtifacts], output_dir: Path) -> str:
    apply_plot_style()
    frame = pd.DataFrame(
        [
            {
                "Model": model.name,
                "F1-score": model.metrics["f1"],
                "Precision": model.metrics["precision"],
                "Recall": model.metrics["recall"],
                "ROC-AUC": model.metrics["roc_auc"],
            }
            for model in models
        ]
    ).set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    frame.plot(
        kind="bar",
        ax=ax,
        color=["#b85c38", "#1f7a58", "#244c7d", "#d2a679"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Indian vs Global Model Performance", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=8)
    return save_plot(fig, output_dir / "model_comparison.png")


def plot_confusion_matrix(conf_matrix: np.ndarray, output_dir: Path) -> str:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#b85c38", as_cmap=True),
        cbar=False,
        ax=ax,
    )
    ax.set_title("Indian Model Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=0)
    return save_plot(fig, output_dir / "confusion_matrix.png")


def plot_feature_importance(
    feature_importance: list[dict[str, float | str]], output_dir: Path
) -> str:
    apply_plot_style()
    frame = pd.DataFrame(feature_importance).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(frame["label"], frame["importance"], color="#b85c38")
    ax.set_title("Top Drivers In Indian Behavioral Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return save_plot(fig, output_dir / "feature_importance.png")


def bundle_for_json(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": bundle["model_version"],
        "system_name": bundle["system_name"],
        "dataset_meta": bundle["dataset_meta"],
        "fusion_weights": bundle["fusion_weights"],
        "metrics": bundle["metrics"],
        "dataset_summary": bundle["dataset_summary"],
        "visuals": bundle["visuals"],
        "feature_importance": bundle["feature_importance"],
        "confusion_matrix": bundle["confusion_matrix"],
        "scenarios": bundle["scenarios"],
        "form_options": bundle["form_options"],
    }
