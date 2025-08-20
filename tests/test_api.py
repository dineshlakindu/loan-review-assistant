
import os
os.environ["UNIT_TEST"] = "1"

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200

def test_review_rules_reject_on_low_score():
    payload = {
        "customer_id": "T-LOW",            # any string ok
        "age": 30,
        "employment_status": "employed",
        "income_monthly": 80000.0,
        "debts_monthly": 10000.0,
        "amount": 500000.0,
        "term_months": 24,
        "credit_score": 480,               # < 500 → reject by policy
        "purpose": "personal"
    }
    r = client.post("/review", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] == "Reject"
    assert any("credit score" in msg.lower() for msg in data.get("reasons", []))

def test_review_explain_autofill_logs():
    # count current decisions
    r0 = client.get("/decisions")
    assert r0.status_code == 200
    before = len(r0.json().get("items", []))

    # run review+explain with CSV autofill
    body = {"customer_id": "10001", "amount": 1200000, "term_months": 60, "purpose": "home"}
    r1 = client.post("/review+explain/autofill", json=body)
    assert r1.status_code == 200
    data = r1.json()
    for k in ["decision", "risk_score", "reasons"]:
        assert k in data

    # decisions should increase by at least one
    r2 = client.get("/decisions")
    assert r2.status_code == 200
    after = len(r2.json().get("items", []))
    assert after >= before + 1


def test_review_rules_approve_when_strong():
    payload = {
        "customer_id": "10001",   # use an ID your /mock endpoints handle
        "age": 40,
        "employment_status": "employed",
        "income_monthly": 200000.0,
        "debts_monthly": 20000.0,
        "amount": 500000.0,
        "term_months": 36,
        "credit_score": 750,
        "collateral_value": 1000000.0,
        "purpose": "home"
    }
    r = client.post("/review", json=payload)
    assert r.status_code == 200
    assert r.json()["decision"] == "Approve"
