# api/engine/agent_graph.py
from __future__ import annotations

import os, json, requests
from typing import TypedDict, Optional, Dict, Any, List, Tuple

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
AML_FALLBACK_BASE = os.getenv("AML_BASE", "http://127.0.0.1:8001")

class ReviewState(TypedDict, total=False):
    application: Dict[str, Any]
    kyc: Optional[Dict[str, Any]]
    credit: Optional[Dict[str, Any]]
    aml: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    rule_hits: List[Dict[str, Any]]
    decision: str
    confidence: float
    explanation: str
    kyc_status: str                     # "pass" | "fail" | "unknown"
    aml_hit: Optional[bool]             # True | False | None
    result: Dict[str, Any]

# ---------------- HTTP helpers ----------------

def _get_json(url: str, method: str = "GET", payload: Optional[dict] = None) -> Optional[dict]:
    try:
        if method.upper() == "POST":
            r = requests.post(url, json=payload, timeout=5)
        else:
            r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------- Data derivation helpers ----------------

def _kyc_status_from_record(rec: Optional[Dict[str, Any]]) -> str:
    """
    Your mock KYC JSON includes: id_document_valid (bool), address_match_score (0..1),
    aml_risk_score (0..100), pep_flag (bool). There's no 'status' key.
    """
    if not rec:
        return "unknown"
    try:
        ok = (
            bool(rec.get("id_document_valid", True)) and
            float(rec.get("address_match_score", 1.0)) >= 0.70 and
            int(rec.get("aml_risk_score", 100)) < 70 and
            not bool(rec.get("pep_flag", False))
        )
        return "pass" if ok else "fail"
    except Exception:
        return "unknown"

def _aml_hit_from_record(rec: Optional[Dict[str, Any]]) -> Optional[bool]:
    if rec is None:
        return None
    try:
        return bool(rec.get("watchlist_hit", False))
    except Exception:
        return None

def _compute_metrics(app: Dict[str, Any]) -> Tuple[float, Optional[float]]:
    income = float(app.get("income_monthly", 0.0) or 0.0)
    debts = float(app.get("debts_monthly", 0.0) or 0.0)
    amount = float(app.get("amount", 0.0) or 0.0)
    cv_raw = app.get("collateral_value", None)
    collateral_value = float(cv_raw) if (cv_raw is not None and str(cv_raw) != "" and float(cv_raw) > 0) else None

    dti = round(debts / income, 2) if income > 0 else 0.0
    ltv = round(amount / collateral_value, 2) if collateral_value and collateral_value > 0 else None
    return dti, ltv

def _risk_score(app: Dict[str, Any], dti: float, ltv: Optional[float]) -> int:
    score = 50
    # DTI
    if dti >= 1.0: score += 30
    elif dti >= 0.6: score += 18
    elif dti >= 0.4: score += 10
    elif dti >= 0.3: score += 5
    # LTV
    if ltv is not None:
        if ltv > 1.2: score += 25
        elif ltv > 1.0: score += 15
        elif ltv > 0.8: score += 8
    else:
        score += 5  # slight risk without collateral
    # Credit score
    cs = int(app.get("credit_score", 0) or 0)
    if cs < 550: score += 30
    elif cs < 600: score += 20
    elif cs < 650: score += 12
    elif cs < 700: score += 6
    # Employment
    emp = str(app.get("employment_status", "")).lower().strip()
    if emp in {"unemployed", "student"}:
        score += 20
    elif emp in {"contract"}:
        score += 10
    # Age
    if int(app.get("age", 0) or 0) < 21:
        score += 10
    # Amount vs income
    inc = float(app.get("income_monthly", 0) or 0)
    amt = float(app.get("amount", 0) or 0)
    if amt > inc * 20:
        score += 10
    # Purpose
    if str(app.get("purpose", "")).lower() == "personal":
        score += 5

    return max(0, min(100, score))

# ---------------- Graph nodes ----------------

def fetch_external(state: ReviewState) -> ReviewState:
    app = state["application"]
    cid = str(app.get("customer_id"))

    kyc = _get_json(f"{API_BASE}/mock/kyc/{cid}")
    credit = _get_json(f"{API_BASE}/mock/credit/{cid}")

    aml = _get_json(f"{API_BASE}/mock/aml/{cid}")
    if aml is None:
        aml = _get_json(f"{AML_FALLBACK_BASE}/aml_screen", method="POST", payload={"customer_id": cid})

    ns = dict(state)
    ns["kyc"], ns["credit"], ns["aml"] = kyc, credit, aml
    return ns

def score_and_decide(state: ReviewState) -> ReviewState:
    app = state["application"]
    credit = state.get("credit") or {}
    kyc_rec = state.get("kyc") or None
    aml_rec = state.get("aml") or None

    # prefer credit API score if present
    credit_score = float(credit.get("credit_score", app.get("credit_score", 600)))

    dti, ltv = _compute_metrics(app)
    rule_hits: List[Dict[str, Any]] = []

    # Basic rules (align with main.py)
    if credit_score >= 700:
        rule_hits.append({"code": "GOOD_SCORE", "message": "Good credit score (>= 700).", "severity": "info"})
    elif credit_score < 580:
        rule_hits.append({"code": "LOW_SCORE", "message": "Low credit score (< 580).", "severity": "reject"})

    if dti <= 0.40:
        rule_hits.append({"code": "GOOD_DTI", "message": "Acceptable DTI (<= 0.40).", "severity": "info"})
    elif dti >= 0.60:
        rule_hits.append({"code": "HIGH_DTI", "message": "High DTI (>= 0.60).", "severity": "reject"})
    else:
        rule_hits.append({"code": "MID_DTI", "message": "Moderate DTI.", "severity": "warn"})

    if ltv is None:
        rule_hits.append({"code": "NO_COLLATERAL", "message": "No collateral provided.", "severity": "warn"})
    elif ltv <= 0.80:
        rule_hits.append({"code": "GOOD_LTV", "message": "Low LTV (<= 0.80).", "severity": "info"})
    elif ltv > 1.20:
        rule_hits.append({"code": "HIGH_LTV", "message": "High LTV (> 1.20).", "severity": "warn"})

    if float(app.get("income_monthly", 0) or 0) >= 100_000:
        rule_hits.append({"code": "HIGH_INCOME", "message": "High income (>= 100,000).", "severity": "info"})

    # Compliance derivation
    kyc_status = _kyc_status_from_record(kyc_rec)
    aml_hit = _aml_hit_from_record(aml_rec)

    if kyc_status == "fail":
        rule_hits.append({"code": "KYC_FAIL", "message": "KYC failed (mock).", "severity": "reject"})
    if aml_hit is True:
        # only add once; keep as error-level so reasons pick it up prominently
        rule_hits.append({"code": "AML_HIT", "message": "AML watchlist hit (mock).", "severity": "flag"})
    if kyc_status == "unknown" or aml_hit is None:
        rule_hits.append({"code": "COMPLIANCE_UNKNOWN", "message": "Compliance services unavailable; safety flag.", "severity": "warn"})

    # Risk score
    risk = _risk_score(app, dti, ltv)

    # Initial label from rules
    blocking = any(h["severity"] == "reject" for h in rule_hits)
    risky = any(h["severity"] == "warn" or h["severity"] == "flag" for h in rule_hits)
    if blocking:
        decision, confidence = "Reject", 0.95
    elif risky:
        decision, confidence = "Flag", 0.55
    else:
        decision, confidence = "Approve", 0.85

    # Align with main.py compliance overrides
    if kyc_status == "fail":
        decision = "Reject"
        confidence = min(confidence, 0.2)
    elif aml_hit is True and decision != "Reject":
        decision = "Flag"
        confidence = min(confidence, 0.5)
    elif kyc_status == "unknown" or aml_hit is None:
        if decision == "Approve":
            decision = "Flag"
        confidence = min(confidence, 0.6)

    ns = dict(state)
    ns["metrics"] = {"dti": dti, "ltv": ltv, "risk_score": risk}
    ns["rule_hits"] = rule_hits
    ns["decision"] = decision
    ns["confidence"] = round(float(confidence), 2)
    ns["kyc_status"] = kyc_status
    ns["aml_hit"] = aml_hit
    return ns

def explain_with_llm(state: ReviewState) -> ReviewState:
    llm = ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        num_predict=220,
        stop=["\n- ", "\n* ", "\n• ", "\n1. ", "\n2. ", "\n3. "],
    )
    summary = {
        "decision": state["decision"],
        "confidence": state["confidence"],
        "metrics": state.get("metrics", {}),
        "rule_hits": state.get("rule_hits", []),
        "kyc_status": state.get("kyc_status"),
        "aml_hit": state.get("aml_hit"),
        "credit": state.get("credit", {}),
        "application": state["application"],
    }
    prompt = (
        "You are a bank loan officer. Explain the loan decision clearly in one short paragraph. "
        "Avoid bullet points. Be factual and calm.\n\nCONTEXT:\n"
        f"{json.dumps(summary, ensure_ascii=False)}"
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    explanation = (msg.content or "").strip()

    ns = dict(state)
    ns["explanation"] = explanation
    return ns

def finalize(state: ReviewState) -> ReviewState:
    res = {
        "decision": state["decision"],
        "confidence": state["confidence"],
        "dti": state["metrics"]["dti"],
        "ltv": state["metrics"]["ltv"],
        "risk_score": state["metrics"]["risk_score"],
        "reasons": [h["message"] for h in state.get("rule_hits", []) if h.get("severity") in ("warn", "flag", "reject")],
        "rule_hits": state.get("rule_hits", []),
        "explanation": state.get("explanation", ""),
        "kyc_status": state.get("kyc_status", "unknown"),
        "aml_hit": state.get("aml_hit", None),
    }
    ns = dict(state)
    ns["result"] = res
    return ns

def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_external", fetch_external)
    g.add_node("score_and_decide", score_and_decide)
    g.add_node("explain_with_llm", explain_with_llm)
    g.add_node("finalize", finalize)
    g.set_entry_point("fetch_external")
    g.add_edge("fetch_external", "score_and_decide")
    g.add_edge("score_and_decide", "explain_with_llm")
    g.add_edge("explain_with_llm", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

GRAPH = build_graph()

def run_review_with_graph(application_payload: Dict[str, Any]) -> Dict[str, Any]:
    state: ReviewState = {"application": application_payload}
    out = GRAPH.invoke(state)
    return out["result"]
