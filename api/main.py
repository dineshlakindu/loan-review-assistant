# api/main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, Literal, List, Tuple, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from fastapi.encoders import jsonable_encoder
import pandas as pd
import os, requests
import math

# --- env / paths ---
load_dotenv()  # picks up OLLAMA_URL / OLLAMA_MODEL / MOCK_BASE from .env

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
APPLICATIONS_PATH = DATA_DIR / "applications.csv"   # reserved
DECISIONS_LOG = DATA_DIR / "decisions_log.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"

MOCK_BASE = os.getenv("MOCK_BASE", "http://127.0.0.1:8000")

# canonical CSV column order (do not change casually)
DECISION_COLS = [
    "ts", "customer_id", "age", "income_monthly", "debts_monthly",
    "amount", "term_months", "credit_score", "employment_status",
    "collateral_value", "purpose",
    "out_decision", "out_dti", "out_ltv", "out_risk_score",
    "out_reasons", "out_confidence", "out_rule_hits",
    "kyc_status", "aml_hit",
]

# --- app ---
app = FastAPI(title="Loan Review API", version="1.5.3")

# CORS so Streamlit/Gradio on localhost can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501", "http://localhost:8501",
        "http://127.0.0.1:7860", "http://localhost:7860",
        "http://127.0.0.1:3000", "http://localhost:3000",
        "http://127.0.0.1:8502", "http://localhost:8502",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the mock router (KYC/Credit/AML endpoints)
from api.routers.mock_external import router as mock_router
app.include_router(mock_router)

EMPLOYMENT_ALLOWED = {
    "employed","self","government","business",
    "student","retired","contract","unemployed"
}
PURPOSE_ALLOWED = {"home","vehicle","education","personal","business"}

# -------------------------
# Models
# -------------------------
class LoanApplication(BaseModel):
    customer_id: str
    age: int
    income_monthly: float
    debts_monthly: float
    amount: float
    term_months: int
    credit_score: int
    employment_status: Literal[
        "employed","self","government","business","student","retired","contract","unemployed"
    ]
    collateral_value: Optional[float] = None
    purpose: Literal["home","vehicle","education","personal","business"]

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 18 or v > 75:
            raise ValueError("Age must be between 18 and 75")
        return v

    @field_validator("income_monthly", "amount", "term_months")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("credit_score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        if v < 300 or v > 850:
            raise ValueError("credit_score must be in [300, 850]")
        return v

# Permissive payload for CSV-autofill endpoints
class PartialApplication(BaseModel):
    customer_id: str
    amount: float
    term_months: int
    purpose: Literal["home","vehicle","education","personal","business"]
    # optional overrides (if provided, they win over CSV)
    collateral_value: Optional[float] = None
    debts_monthly: Optional[float] = None

    @field_validator("amount", "term_months")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v

# -------------------------
# Customers directory (from CSV)
# -------------------------
def _load_customers_df(safe: bool = False) -> pd.DataFrame:
    """
    Load customers.csv. If safe=True, never raise: return empty DataFrame on issues.
    Expected columns (min): customer_id, age, employment_status, income_monthly, credit_score
    Optionally: debts_monthly, collateral_value.
    """
    try:
        if not CUSTOMERS_CSV.exists():
            if safe:
                return pd.DataFrame()
            raise FileNotFoundError(f"Missing customers CSV at {CUSTOMERS_CSV}")
        return pd.read_csv(CUSTOMERS_CSV)
    except Exception:
        if safe:
            return pd.DataFrame()
        raise

@app.get("/customers", summary="List available customer IDs")
def list_customers() -> List[str]:
    df = _load_customers_df()
    if "customer_id" not in df.columns:
        raise HTTPException(status_code=500, detail="customers.csv missing 'customer_id' column")
    return df["customer_id"].astype(str).tolist()

@app.get("/customers/{customer_id}", summary="Get full customer record by ID")
def get_customer(customer_id: str) -> Dict[str, Any]:
    df = _load_customers_df()
    row = df[df["customer_id"].astype(str) == str(customer_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return row.iloc[0].to_dict()

# -------------------------
# Rules helpers
# -------------------------
def compute_metrics(app_data: LoanApplication) -> Tuple[float, Optional[float]]:
    dti = 0.0 if app_data.income_monthly == 0 else round(app_data.debts_monthly / app_data.income_monthly, 2)
    ltv = None
    if app_data.collateral_value and app_data.collateral_value > 0:
        ltv = round(app_data.amount / app_data.collateral_value, 2)
    return dti, ltv

def risk_score(app_data: LoanApplication, dti: float, ltv: Optional[float]) -> int:
    score = 50  # base risk
    if dti >= 1.0: score += 30
    elif dti >= 0.6: score += 18
    elif dti >= 0.4: score += 10
    elif dti >= 0.3: score += 5
    if ltv is not None:
        if ltv > 1.2: score += 25
        elif ltv > 1.0: score += 15
        elif ltv > 0.8: score += 8
    else:
        score += 5  # slight risk without collateral
    cs = app_data.credit_score
    if cs < 550: score += 30
    elif cs < 600: score += 20
    elif cs < 650: score += 12
    elif cs < 700: score += 6
    if app_data.employment_status in {"unemployed", "student"}:
        score += 20
    elif app_data.employment_status in {"contract"}:
        score += 10
    if app_data.age < 21:
        score += 10
    if app_data.amount > app_data.income_monthly * 20:
        score += 10
    if app_data.purpose == "personal":
        score += 5
    return max(0, min(100, score))

def decision_rules(app_data: LoanApplication, dti: float, ltv: Optional[float], risk: int) -> Tuple[str, List[str], List[dict], float]:
    reasons: List[str] = []
    hits: List[dict] = []

    def add(code, msg, sev="info"):
        hits.append({"code": code, "message": msg, "severity": sev})

    # Hard fails
    if app_data.credit_score < 500:
        msg = "Very low credit score (< 500)."
        reasons.append(msg); add("LOW_SCORE", msg, "reject")
    if app_data.income_monthly < 50000:
        msg = "Low income (< 50,000)."
        reasons.append(msg); add("LOW_INCOME", msg, "flag")
    if app_data.employment_status == "unemployed":
        msg = "Unemployed status."
        reasons.append(msg); add("UNEMPLOYED", msg, "reject")
    if dti >= 0.6:
        msg = "High DTI (>= 0.6)."
        reasons.append(msg); add("HIGH_DTI", msg, "reject")
    if ltv is not None and ltv > 1.2:
        msg = "High LTV (> 1.2)."
        reasons.append(msg); add("HIGH_LTV", msg, "flag")

    if reasons:
        label = "Reject" if ("High DTI (>= 0.6)." in reasons or app_data.credit_score < 500) else "Flag"
        confidence = max(0.05, min(0.95, 1 - risk/100))
        return label, reasons, hits, round(confidence, 2)

    positives = []
    if app_data.credit_score >= 700:
        positives.append("Good credit score (>= 700)."); add("GOOD_SCORE", positives[-1], "info")
    if dti <= 0.4:
        positives.append("Acceptable DTI (<= 0.4)."); add("GOOD_DTI", positives[-1], "info")
    if ltv is None or ltv <= 0.8:
        positives.append("Low LTV (<= 0.8) or no collateral needed."); add("GOOD_LTV", positives[-1], "info")
    if app_data.income_monthly >= 100000:
        positives.append("High income (>= 100,000)."); add("HIGH_INCOME", positives[-1], "info")

    if risk <= 65 and len(positives) >= 2:
        confidence = max(0.6, min(0.95, 1 - risk/100))
        return "Approve", positives, hits, round(confidence, 2)

    add("BORDERLINE", "Borderline metrics; needs manual review.", "warn")
    confidence = max(0.3, min(0.8, 1 - risk/100))
    return "Flag", ["Borderline metrics; needs manual review."], hits, round(confidence, 2)

# --- timestamp helper (UTC Z-suffix) ---
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log_decision(app_data: LoanApplication, result: dict) -> None:
    # write ONLY a single canonical UTC timestamp; column order fixed
    row = {
        "ts": _utc_now_iso(),
        "customer_id": app_data.customer_id,
        "age": app_data.age,
        "income_monthly": app_data.income_monthly,
        "debts_monthly": app_data.debts_monthly,
        "amount": app_data.amount,
        "term_months": app_data.term_months,
        "credit_score": app_data.credit_score,
        "employment_status": app_data.employment_status,
        "collateral_value": app_data.collateral_value,
        "purpose": app_data.purpose,
        "out_decision": result["decision"],
        "out_dti": result["dti"],
        "out_ltv": result["ltv"],
        "out_risk_score": result["risk_score"],
        "out_reasons": str(result["reasons"]),
        "out_confidence": result.get("confidence"),
        "out_rule_hits": str(result.get("rule_hits")),
        "kyc_status": result.get("kyc_status"),
        "aml_hit": result.get("aml_hit"),
    }
    try:
        header_needed = not DECISIONS_LOG.exists() or DECISIONS_LOG.stat().st_size == 0
        pd.DataFrame([row], columns=DECISION_COLS).to_csv(
            DECISIONS_LOG, mode="a", header=header_needed, index=False
        )
    except Exception:
        # never crash the API if logging fails
        pass

# -------------------------
# Compliance helpers
# -------------------------
def check_kyc(customer_id: str):
    # Short-circuit in tests to avoid downgrading decisions
    if os.getenv("UNIT_TEST") == "1":
        return "pass"
    try:
        r = requests.get(f"{MOCK_BASE}/mock/kyc/{customer_id}", timeout=3)
        if r.status_code != 200:
            return "unknown"
        rec = r.json()
        ok = (
            rec.get("id_document_valid", True)
            and float(rec.get("address_match_score", 1.0)) >= 0.70
            and int(rec.get("aml_risk_score", 100)) < 70
            and not bool(rec.get("pep_flag", False))
        )
        return "pass" if ok else "fail"
    except Exception:
        return "unknown"

def check_aml(customer_id: str):
    """Query the AML endpoint."""
    # Short-circuit in tests to avoid downgrading decisions
    if os.getenv("UNIT_TEST") == "1":
        return False
    try:
        r = requests.get(f"{MOCK_BASE}/mock/aml/{customer_id}", timeout=3)
        if r.status_code != 200:
            return None
        return bool(r.json().get("watchlist_hit", False))
    except Exception:
        return None

# -------------------------
# Core review runner (used by multiple endpoints)
# -------------------------
def _run_review(app_data: LoanApplication) -> Dict[str, Any]:
    dti, ltv = compute_metrics(app_data)
    risk = risk_score(app_data, dti, ltv)

    decision, reasons, rule_hits, confidence = decision_rules(app_data, dti, ltv, risk)

    kyc_status = check_kyc(app_data.customer_id)
    aml_hit = check_aml(app_data.customer_id)

    if kyc_status == "fail":
        decision = "Reject"
        reasons = ["KYC failed (mock)."] + reasons
        confidence = min(confidence, 0.2)
        rule_hits.append({"code": "KYC_FAIL", "message": "KYC failed (mock).", "severity": "reject"})
    elif aml_hit is True and decision != "Reject":
        decision = "Flag"
        reasons = ["AML watchlist hit (mock)."] + reasons
        confidence = min(confidence, 0.5)
        rule_hits.append({"code": "AML_HIT", "message": "AML watchlist hit (mock).", "severity": "flag"})
    elif kyc_status == "unknown" or aml_hit is None:
        if decision == "Approve":
            decision = "Flag"
            reasons = ["Compliance services unavailable; safety flag."] + reasons
            rule_hits.append({"code": "COMPLIANCE_UNKNOWN", "message": "Compliance services unavailable; safety flag.", "severity": "warn"})
        confidence = min(confidence, 0.6)

    result = {
        "decision": decision,
        "dti": dti,
        "ltv": ltv,
        "risk_score": risk,
        "reasons": reasons,
        "rule_hits": rule_hits,
        "confidence": round(confidence, 2),
        "kyc_status": kyc_status,
        "aml_hit": aml_hit,
    }
    return result

# -------------------------
# Autofill from customers.csv
# -------------------------
def _autofill_to_full_app(p: PartialApplication) -> LoanApplication:
    """
    Builds a full LoanApplication by reading customers.csv for the given customer_id
    and merging with the minimal fields provided in PartialApplication.
    If some fields are missing in the CSV, we apply reasonable defaults.
    """
    df = _load_customers_df(safe=True)
    df.columns = [c.lower() for c in df.columns]

    row = df[df.get("customer_id", pd.Series([], dtype=str)).astype(str) == str(p.customer_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {p.customer_id} not found in customers.csv")
    r = row.iloc[0].to_dict()

    # required from CSV
    try:
        age = int(r.get("age"))
        employment_status = str(r.get("employment_status")).strip().lower()
        income_monthly = float(r.get("income_monthly"))
        credit_score = int(r.get("credit_score"))
    except Exception:
        raise HTTPException(status_code=422, detail="customers.csv is missing required fields for this customer")

    # optional from CSV
    debts_monthly = p.debts_monthly
    if debts_monthly is None:
        dm = r.get("debts_monthly")
        debts_monthly = float(dm) if dm is not None and str(dm) != "" else 0.0

    collateral_value = p.collateral_value
    if collateral_value is None:
        cv = r.get("collateral_value")
        if cv is not None and str(cv) != "":
            try:
                collateral_value = float(cv)
            except Exception:
                collateral_value = None

    if employment_status not in EMPLOYMENT_ALLOWED:
        raise HTTPException(status_code=422, detail=f"employment_status '{employment_status}' not in allowed set")

    return LoanApplication(
        customer_id=str(p.customer_id),
        age=age,
        income_monthly=income_monthly,
        debts_monthly=debts_monthly,
        amount=float(p.amount),
        term_months=int(p.term_months),
        credit_score=credit_score,
        employment_status=employment_status,
        collateral_value=collateral_value,
        purpose=p.purpose,
    )

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Loan Review API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/config")
def config():
    """Expose useful runtime config for the UI."""
    return {
        "mock_base": MOCK_BASE,
        "ollama_url": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "tinyllama"),
    }

# Original endpoint (expects FULL payload)
@app.post("/review")
def review_application(app_data: LoanApplication):
    result = _run_review(app_data)
    log_decision(app_data, result)
    return result

# Safer decisions with limit + never crash
@app.get("/decisions")
def list_decisions(limit: int = Query(default=50, ge=1, le=1000)):
    """
    Return the last N decisions from data/decisions_log.csv.
    Robust to partial/locked CSVs and embedded commas/quotes.
    Always returns {"items": [...]} with JSON-safe values (no NaN/Inf).
    """
    try:
        if not DECISIONS_LOG.exists() or DECISIONS_LOG.stat().st_size == 0:
            return {"items": [], "note": "no decisions yet"}

        # Forgiving parse (skips malformed lines if any)
        try:
            df = pd.read_csv(DECISIONS_LOG, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(DECISIONS_LOG, engine="python", on_bad_lines="skip")

        if df.empty:
            return {"items": [], "note": "log is empty"}

        # Parse datetimes if present and sort by ts (fallback: ts_local)
        sort_col = None
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            sort_col = "ts"
        if "ts_local" in df.columns:
            df["ts_local"] = pd.to_datetime(df["ts_local"], errors="coerce")
            if sort_col is None:
                sort_col = "ts_local"
        if sort_col:
            df = df.sort_values(sort_col)

        if len(df) > limit:
            df = df.tail(limit)

        # Make datetimes pretty strings (so we don't return numpy datetime types)
        if "ts" in df.columns:
            df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "ts_local" in df.columns:
            df["ts_local"] = df["ts_local"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # JSON-safe: replace NaN/Inf with None
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.where(pd.notnull(df), None)

        # Build JSON-safe records (catch any lingering float NaN)
        records = []
        for rec in df.to_dict(orient="records"):
            clean = {}
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    clean[k] = None
                else:
                    clean[k] = v
            records.append(clean)

        # Extra safety for datetimes / numpy types
        return {"items": jsonable_encoder(records)}

    except Exception as e:
        # Last-resort fallback with csv module
        try:
            import csv, io
            with open(DECISIONS_LOG, "r", encoding="utf-8", newline="") as f:
                text = f.read()
            rows = list(csv.DictReader(io.StringIO(text)))
            if len(rows) > limit:
                rows = rows[-limit:]
            # Best-effort: normalize empty strings to None
            items = [{k: (v if v not in ("", "NaN", "nan") else None) for k, v in r.items()} for r in rows]
            return {"items": items, "note": f"csv fallback: {e!s}"}
        except Exception as e2:
            return {"items": [], "note": f"decisions_log not readable: {e2!s}"}

# LLM explanation (kept as-is for your UI)
@app.post("/review+explain")
def review_and_explain(app_data: LoanApplication):
    result = _run_review(app_data)
    try:
        from api.engine.agent import llm_explain_structured
        pack = llm_explain_structured(result)
        result["llm_explanation"] = pack.get("explanation", "")
        result.setdefault("confidence", pack.get("confidence", result["confidence"]))
    except Exception:
        from api.engine.agent import llm_explain
        result["llm_explanation"] = llm_explain(result)
    log_decision(app_data, result)
    return result

# ---- CSV-autofill endpoints ----
@app.post("/review/autofill")
def review_autofill(p: PartialApplication):
    full = _autofill_to_full_app(p)
    res = _run_review(full)
    log_decision(full, res)
    return res

@app.post("/review+explain/autofill")
def review_and_explain_autofill(p: PartialApplication):
    full = _autofill_to_full_app(p)
    res = _run_review(full)
    try:
        from api.engine.agent import llm_explain_structured
        pack = llm_explain_structured(res)
        res["llm_explanation"] = pack.get("explanation", "")
        res.setdefault("confidence", pack.get("confidence", res["confidence"]))
    except Exception:
        from api.engine.agent import llm_explain
        res["llm_explanation"] = llm_explain(res)
    log_decision(full, res)
    return res

# ---------- LangGraph agent router ----------
from api.engine.agent_graph import run_review_with_graph
graph_router = APIRouter(prefix="/graph", tags=["Agent (LangGraph)"])

@graph_router.post("/review", summary="Run loan review via LangGraph agent")
def review_via_graph(app_data: LoanApplication):
    result = run_review_with_graph(app_data.model_dump())
    # ensure decisions_log.csv gets a row even with LangGraph ON
    try:
        log_decision(app_data, result)
    except Exception:
        pass
    return result

app.include_router(graph_router)
