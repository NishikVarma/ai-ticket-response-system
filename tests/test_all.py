import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["TICKET_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest
from fastapi.testclient import TestClient

import db
import ml
import rag
from app import app

client = TestClient(app, follow_redirects=True)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Default: LLM works and echoes the retrieved context, so tests don't need Ollama."""
    monkeypatch.setattr(rag, "generate", lambda t, s, p: "DRAFT based on: " + " | ".join(x[1] for x in p))


def make_ticket(**over):
    data = {"customer_name": "Ann", "subject": "Unable to log in",
            "description": "I forgot my password and cannot access my account.",
            "category": "Account", "priority": "High"}
    data.update(over)
    r = client.post("/tickets", data=data)
    return r


def last_id():
    return db.list_tickets()[0]["id"]


def test_ml_predicts_for_different_ticket_types():
    cases = {
        ("Unable to log in", "I forgot my password and cannot access my account."): "Reset the password",
        ("Charged twice", "I see two payments for one order."): "Refund the duplicate charge",
        ("App crashes", "The app closes when I open it."): "Update to the latest app version",
        ("Need invoice", "Send me a receipt for my payment."): "Download the invoice",
    }
    for (s, d), expected in cases.items():
        solution, conf = ml.predict(s, d)
        assert solution.startswith(expected), (s, solution)


def test_rag_retrieves_relevant_kb():
    hits = rag.retrieve("Unable to log in. forgot my password. Reset the password using the account recovery process.")
    assert hits and {h[0] for h in hits} == {"password_reset.md"}   # no unrelated documents
    assert any("Forgot Password" in h[1] for h in hits)


def test_crud_and_ai_pipeline():
    r = make_ticket()
    assert r.status_code == 200
    t, _ = db.get_ticket(last_id())
    assert t["predicted_solution"].startswith("Reset the password")
    assert "Forgot Password" in t["ai_response"]          # KB content reached the LLM input
    # list + detail
    assert "Unable to log in" in client.get("/").text
    # update status + edited final response
    client.post(f"/tickets/{t['id']}", data={"customer_name": "Ann", "subject": "Unable to log in",
                "description": "x", "category": "Account", "priority": "High",
                "status": "Resolved", "final_response": "Edited reply"})
    t, _ = db.get_ticket(t["id"])
    assert t["status"] == "Resolved" and t["final_response"] == "Edited reply"
    # soft delete
    client.post(f"/tickets/{t['id']}/delete")
    assert db.get_ticket(t["id"])[0] is None


def test_missing_required_fields():
    before = len(db.list_tickets())
    r = make_ticket(subject="", customer_name="")
    assert "Required" in r.text and len(db.list_tickets()) == before


def test_dynamic_fields():
    client.post("/fields", data={"label": "Order number", "type": "number", "required": "1"})
    client.post("/fields", data={"label": "Plan", "type": "dropdown", "options": "Free, Pro"})
    client.post("/fields", data={"label": "Is VIP", "type": "checkbox"})
    page = client.get("/tickets/new").text
    assert "Order number" in page and "Pro" in page and "Is VIP" in page
    fid = {f["label"]: f["id"] for f in db.active_fields()}
    # required number missing / invalid
    assert "Required" in make_ticket().text
    assert "Invalid number" in make_ticket(**{f"field_{fid['Order number']}": "abc"}).text
    assert "Choose a listed option" in make_ticket(**{f"field_{fid['Order number']}": "5", f"field_{fid['Plan']}": "Gold"}).text
    # valid
    make_ticket(**{f"field_{fid['Order number']}": "42", f"field_{fid['Plan']}": "Pro", f"field_{fid['Is VIP']}": "1"})
    _, vals = db.get_ticket(last_id())
    assert vals[fid["Order number"]] == "42" and vals[fid["Plan"]] == "Pro" and vals[fid["Is VIP"]] == "1"
    for f in db.active_fields():
        db.deactivate_field(f["id"])


def test_llm_failure_is_graceful(monkeypatch):
    def boom(*a):
        raise rag.LLMError("Local LLM unavailable")
    monkeypatch.setattr(rag, "generate", boom)
    assert make_ticket().status_code == 200
    t, _ = db.get_ticket(last_id())
    assert t["predicted_solution"] and t["ai_response"] == "" and "unavailable" in t["ai_error"]
    # regenerate after the LLM recovers
    monkeypatch.setattr(rag, "generate", lambda *a: "Recovered reply")
    client.post(f"/tickets/{t['id']}/regenerate")
    t, _ = db.get_ticket(t["id"])
    assert t["ai_response"] == "Recovered reply" and t["ai_error"] == ""


def test_missing_model_is_graceful(monkeypatch):
    def nomodel(*a):
        raise FileNotFoundError("model.joblib")
    monkeypatch.setattr(ml, "predict", nomodel)
    assert make_ticket().status_code == 200
    t, _ = db.get_ticket(last_id())
    assert "ML prediction failed" in t["ai_error"]


def test_irrelevant_ticket_gets_fallback_not_llm(monkeypatch):
    monkeypatch.setattr(ml, "predict", lambda s, d: ("Reset the password using the account recovery process.", 0.9))
    monkeypatch.setattr(rag, "retrieve", lambda q: [])
    make_ticket()
    t, _ = db.get_ticket(last_id())
    assert "support team" in t["ai_response"] and "No relevant" in t["ai_error"]


def test_signature_and_placeholders_removed():
    raw = "Hello,\n\nUse the link.\n\nBest regards,\n[Your Company] Support Team"
    assert rag.clean(raw) == "Hello,\n\nUse the link."
