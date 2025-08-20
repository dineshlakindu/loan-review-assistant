from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

app = FastAPI(title="Mock KYC/AML API", version="1.1.0")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KYC_PATH = DATA_DIR / "kyc.csv"
AML_PATH = DATA_DIR / "aml.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"

# Load once at startup
if not (KYC_PATH.exists() and AML_PATH.exists()):
    raise RuntimeError("kyc.csv / aml.csv not found in data/")

kyc_df = pd.read_csv(KYC_PATH)           # columns: customer_id, kyc_status
aml_df = pd.read_csv(AML_PATH)           # columns: customer_id, watchlist_hit
customers_df = pd.read_csv(CUSTOMERS_PATH)  # optional lookup

class KycRequest(BaseModel):
    customer_id: int

class AmlRequest(BaseModel):
    customer_id: int

@app.post("/kyc_check")
def kyc_check(req: KycRequest):
    row = kyc_df.loc[kyc_df["customer_id"] == req.customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="customer_id not found")
    return {"customer_id": req.customer_id, "kyc_status": row.iloc[0]["kyc_status"]}

@app.post("/aml_screen")
def aml_screen(req: AmlRequest):
    row = aml_df.loc[aml_df["customer_id"] == req.customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="customer_id not found")
    return {"customer_id": req.customer_id, "watchlist_hit": bool(row.iloc[0]["watchlist_hit"])}
