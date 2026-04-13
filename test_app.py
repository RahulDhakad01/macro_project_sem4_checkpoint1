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
