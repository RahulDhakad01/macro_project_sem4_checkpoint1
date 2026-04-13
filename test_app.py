from app import create_app


def test_health_endpoint():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["selected_model"]


def test_prediction_endpoint():
    app = create_app()
    client = app.test_client()

    payload = {
        "amount": 2400,
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
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert "fraud_probability" in body
    assert len(body["top_factors"]) > 0
