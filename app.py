from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from fraud_model import FraudDetectionService


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"


def create_app() -> Flask:
    app = Flask(__name__)
    service = FraudDetectionService(model_dir=MODEL_DIR)
    service.ensure_ready()
    app.config["FRAUD_SERVICE"] = service

    @app.get("/")
    def index():
        payload = service.dashboard_payload()
        return render_template("index.html", dashboard=payload)

    @app.get("/api/health")
    def health():
        model_summary = service.model_summary()
        return jsonify(
            {
                "status": "ok",
                "selected_model": model_summary["name"],
                "f1_score": model_summary["metrics"]["f1"],
            }
        )

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or request.form.to_dict()

        try:
            prediction = service.predict(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        return jsonify({"status": "ok", **prediction})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
