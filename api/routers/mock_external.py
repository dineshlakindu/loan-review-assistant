from fastapi import APIRouter, HTTPException
from api.models.schemas import KYCRecord, CreditReport
from api.services.mock_providers import get_kyc, get_credit

# NEW: for AML CSV reading
from pathlib import Path
import pandas as pd

router = APIRouter(prefix="/mock", tags=["mock"])

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AML_CSV = DATA_DIR / "aml.csv"  # expects columns: customer_id, watchlist_hit

@router.get("/kyc/{customer_id}", response_model=KYCRecord)
def mock_kyc(customer_id: str):
    rec = get_kyc(customer_id)
    if not rec:
        raise HTTPException(status_code=404, detail="KYC record not found")

    # Your CSV has only: customer_id, kyc_status (pass|fail)
    status = str(rec.get("kyc_status", "pass")).strip().lower()  # pass by default
    is_pass = status == "pass"

    # Construct a full KYCRecord from that single flag
    normalized = KYCRecord(
        customer_id=str(rec.get("customer_id", customer_id)),
        watchlist_hit=False if is_pass else True,          # fail -> simulate hit
        pep_flag=False,                                    # keep simple
        id_document_valid=True if is_pass else False,
        address_match_score=0.92 if is_pass else 0.55,
        aml_risk_score=12 if is_pass else 75,
        sanctions_sources=[] if is_pass else ["MOCK"]
    )
    return normalized

@router.get("/credit/{customer_id}", response_model=CreditReport)
def mock_credit(customer_id: str):
    rec = get_credit(customer_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Credit report not found")

    # Your credit.csv already has the needed columns
    return CreditReport(
        customer_id=str(rec.get("customer_id", customer_id)),
        credit_score=int(float(rec.get("credit_score", 650))),
        delinquencies_12m=int(float(rec.get("delinquencies_12m", 0))),
        utilization_ratio=float(rec.get("utilization_ratio", 0.2)),
        inquiries_6m=int(float(rec.get("inquiries_6m", 0))),
        credit_limit_total=float(rec.get("credit_limit_total", 0)),
        on_time_payment_rate=float(rec.get("on_time_payment_rate", 0.95)),
    )

# NEW: AML mock (integrated)
@router.get("/aml/{customer_id}")
def mock_aml(customer_id: str):
    """
    Returns: { "customer_id": "...", "watchlist_hit": bool }
    Safe defaults if CSV missing/corrupt or row not found.
    """
    try:
        if not AML_CSV.exists():
            return {"customer_id": customer_id, "watchlist_hit": False}
        df = pd.read_csv(AML_CSV, dtype=str).fillna("")
        row = df.loc[df["customer_id"].astype(str) == str(customer_id)]
        if row.empty:
            return {"customer_id": customer_id, "watchlist_hit": False}
        raw = str(row.iloc[0].get("watchlist_hit", "")).strip().lower()
        truthy = {"true", "1", "yes", "y"}
        return {"customer_id": customer_id, "watchlist_hit": raw in truthy}
    except Exception:
        return {"customer_id": customer_id, "watchlist_hit": False}
