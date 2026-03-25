import main


class StubModel:
    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


def test_predict_success():
    main._model = StubModel()
    client = main.app.test_client()

    payload = {
        "Income": 80000,
        "Credit_Score": 720,
        "Loan_Amount": 25000,
        "DTI_Ratio": 18.5,
        "Employment_Status": "employed",
    }
    response = client.post("/predict", json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert data["ok"] is True
    assert data["decision"] == "Approved"


def test_predict_validation_error():
    main._model = StubModel()
    client = main.app.test_client()

    payload = {
        "Income": 80000,
        "Credit_Score": 200,
        "Loan_Amount": 25000,
        "DTI_Ratio": 18.5,
        "Employment_Status": "self-employed",
    }
    response = client.post("/predict", json=payload)
    data = response.get_json()

    assert response.status_code == 400
    assert data["ok"] is False
    assert any("Credit_Score" in message for message in data["errors"])
    assert any("Employment_Status" in message for message in data["errors"])
