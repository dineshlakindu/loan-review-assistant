from pathlib import Path
import csv
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]      # project root
DATA_DIR = ROOT / "data"

def _load_csv_index(fname: str, key: str = "customer_id") -> Dict[str, Any]:
    with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if fname == "kyc.csv":
        for r in rows:
            # your kyc.csv is just customer_id,kyc_status
            r["kyc_status"] = str(r.get("kyc_status", "pass")).strip().lower()
    elif fname == "customers.csv":
        for r in rows:
            # ensure credit_score is numeric
            r["credit_score"] = int(float(r.get("credit_score", 650)))
    return {str(r[key]): r for r in rows}

_KYC = None
_CREDIT = None

def get_kyc(customer_id: str) -> Optional[dict]:
    global _KYC
    if _KYC is None:
        _KYC = _load_csv_index("kyc.csv")
    return _KYC.get(str(customer_id))

def get_credit(customer_id: str) -> Optional[dict]:
    """
    Read credit info from customers.csv (since credit.csv was removed).
    We only need credit_score; other fields will be defaulted in the router.
    """
    global _CREDIT
    if _CREDIT is None:
        _CREDIT = _load_csv_index("customers.csv")
    return _CREDIT.get(str(customer_id))
