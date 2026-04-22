from app import create_app


def test_health_endpoint():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["selected_model"]


def test_metrics_page():
    app = create_app()
    client = app.test_client()

    response = client.get("/metrics")

    assert response.status_code == 200
    assert b"Machine learning evidence" in response.data


def test_about_page():
    app = create_app()
    client = app.test_client()

    response = client.get("/about")

    assert response.status_code == 200
    assert b"Project overview" in response.data


def test_prediction_endpoint():
    app = create_app()
    client = app.test_client()

    payload = {
        "transaction_amount": 8800,
        "transaction_hour": 2,
        "location": "Delhi",
        "merchant_category": "Digital",
        "card_type": "Visa",
        "transactions_last_24h": 12,
        "previous_declined_transactions": 3,
        "distance_from_home": 950,
        "foreign_transaction": 1,
        "card_present": 0,
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert "fraud_score" in body
    assert "P_indian" in body
    assert "P_global" in body
    assert len(body["reasons"]) > 0


def test_low_risk_payload_does_not_overfire_global_model():
    app = create_app()
    client = app.test_client()

    payload = {
        "transaction_amount": 2400,
        "transaction_hour": 15,
        "location": "Pune",
        "merchant_category": "POS",
        "card_type": "Visa",
        "transactions_last_24h": 2,
        "previous_declined_transactions": 0,
        "distance_from_home": 6,
        "foreign_transaction": 0,
        "card_present": 1,
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["risk_level"] != "HIGH"
    assert body["P_global"] < 0.5


def test_builtin_safe_purchase_stays_low_risk():
    app = create_app()
    client = app.test_client()

    payload = {
        "transaction_amount": 2400,
        "transaction_hour": 15,
        "location": "Pune",
        "merchant_category": "POS",
        "card_type": "Visa",
        "transactions_last_24h": 2,
        "previous_declined_transactions": 0,
        "distance_from_home": 6,
        "foreign_transaction": 0,
        "card_present": 1,
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["risk_level"] == "LOW"
    assert body["fraud_score"] < 20


def test_builtin_digital_anomaly_stays_high_risk():
    app = create_app()
    client = app.test_client()

    payload = {
        "transaction_amount": 8600,
        "transaction_hour": 2,
        "location": "Delhi",
        "merchant_category": "Digital",
        "card_type": "MasterCard",
        "transactions_last_24h": 11,
        "previous_declined_transactions": 3,
        "distance_from_home": 940,
        "foreign_transaction": 1,
        "card_present": 0,
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["risk_level"] == "HIGH"
    assert body["fraud_score"] > 80


def test_mixed_signal_case_lands_in_medium_band():
    app = create_app()
    client = app.test_client()

    payload = {
        "transaction_amount": 6500,
        "transaction_hour": 1,
        "location": "Delhi",
        "merchant_category": "Digital",
        "card_type": "Visa",
        "transactions_last_24h": 10,
        "previous_declined_transactions": 2,
        "distance_from_home": 120,
        "foreign_transaction": 0,
        "card_present": 0,
    }

    response = client.post("/api/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["risk_level"] == "MEDIUM"
    assert 30 <= body["fraud_score"] <= 70
