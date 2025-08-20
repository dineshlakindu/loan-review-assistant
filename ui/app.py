# ui/app.py
import os
import math
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# -------------------- Page + theme --------------------
API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Loan Review Assistant", layout="wide")

# Global CSS (subtle colors, bigger decision pill, bigger LLM heading)
st.markdown(
    """
    <style>
        :root{
            --acc:#6ae38f; --acc2:#6bc3ff; --bgc:#0e1117; --card:#10151c;
            --line:rgba(255,255,255,.08); --muted:rgba(255,255,255,.65);
        }
        #MainMenu, header {visibility:hidden;}
        .block-container {padding-top: 2.8rem !important; padding-bottom: 2rem;}
        .titlebar{
            padding:14px 18px; margin:0 0 14px 0;
            background:linear-gradient(90deg, rgba(106,227,143,.10), rgba(107,195,255,.08));
            border:1px solid var(--line); border-radius:14px;
        }
        .titlebar h1{margin:0; font-weight:800;}
        .titlebar p{margin:4px 0 0 0; color:var(--muted)}
        .card{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px;}
        .muted{color:var(--muted); font-size:13px;}
        /* Decision pill — BIGGER */
        .pill{
            display:inline-block; padding:8px 18px; border-radius:999px;
            font-weight:800; font-size:22px; letter-spacing:.3px; text-transform:uppercase;
        }
        .pill-Approve{background:#10381f;color:#6ae38f;}
        .pill-Flag{background:#3c2f17;color:#f3c969;}
        .pill-Reject{background:#3a1d1d;color:#ff8e8e;}
        /* Tighter st.metric */
        .tight > div[data-testid="stMetricValue"]{font-size:28px;}
        .tight > div[data-testid="stMetricDelta"]{font-size:12px;}
        /* Buttons */
        .stButton > button{
            border-radius:10px; border:1px solid var(--line);
            background:linear-gradient(180deg, rgba(107,195,255,.16), rgba(106,227,143,.14));
        }
        .stButton > button:hover{filter:brightness(1.05);}
        /* Bigger "LLM Explanation" title */
        .section-title{
            font-size:28px; font-weight:800; margin:8px 0 6px 0;
            line-height:1.15;
        }
        .label{font-weight:600}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<div class="titlebar"><h1>🧾 Loan Review Assistant</h1>'
    '<p>Enter or autofill an application, then review.</p></div>',
    unsafe_allow_html=True,
)

# -------------------- Session defaults --------------------
if "API_BASE" not in st.session_state:
    st.session_state["API_BASE"] = API_BASE_DEFAULT
if "USE_GRAPH" not in st.session_state:
    st.session_state["USE_GRAPH"] = True

def api_base() -> str:
    return st.session_state.get("API_BASE", API_BASE_DEFAULT)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Settings")
    st.text_input("API", key="API_BASE", help="FastAPI base URL")
    st.toggle("LangGraph", key="USE_GRAPH")
    st.caption("Keep FastAPI on 127.0.0.1:8000")

# -------------------- Helpers --------------------
def _num_or_none(x):
    """Return float(x) if finite & not NaN; otherwise None."""
    try:
        f = float(x)
        if math.isfinite(f) and not pd.isna(f):
            return f
    except Exception:
        pass
    return None

def _int_or_default(x, default):
    try:
        if x is None or (isinstance(x, float) and (pd.isna(x) or not math.isfinite(x))):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

# -------------------- Data access --------------------
def load_customers():
    try:
        return requests.get(f"{api_base()}/customers", timeout=5).json()
    except Exception:
        try:
            df = pd.read_csv("data/customers.csv")
            return df["customer_id"].astype(str).tolist()
        except Exception:
            return []

def get_customer_record(cid: str):
    try:
        r = requests.get(f"{api_base()}/customers/{cid}", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    try:
        df = pd.read_csv("data/customers.csv")
        row = df[df["customer_id"].astype(str) == str(cid)]
        return row.iloc[0].to_dict() if not row.empty else None
    except Exception:
        return None

def get_latest_application_for_customer(cid: str):
    try:
        df = pd.read_csv("data/applications.csv")
        if "customer_id" not in df.columns:
            return None
        rows = df[df["customer_id"].astype(str) == str(cid)]
        if rows.empty:
            return None
        sort_col = "application_id" if "application_id" in rows.columns else rows.columns[0]
        return rows.sort_values(sort_col).iloc[-1].to_dict()
    except Exception:
        return None

# -------------------- Autofill defaults --------------------
EMPLOYMENT_OPTIONS = ["employed","self","government","business","student","retired","contract","unemployed"]
PURPOSE_OPTIONS = ["home","vehicle","education","personal","business"]

DEFAULTS = {
    "age": 26,
    "employment_status": "employed",
    "income_monthly": 150000.0,
    "credit_score": 700,
    "amount": 500000.0,
    "term_months": 24,
    "debts_monthly": 30000.0,
    "collateral_value": 0.0,
    "purpose": "personal",
}

def reset_form_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

def apply_autofill_from_csv(cid: str, overwrite: bool = True):
    """Populate fields from customers.csv + applications.csv."""
    cust = get_customer_record(cid)
    app_ = get_latest_application_for_customer(cid)

    if overwrite:
        reset_form_defaults()

    def setk(k, v):
        if v is None:
            return
        if overwrite or k not in st.session_state or st.session_state.get(k) in (None, "", 0):
            st.session_state[k] = v

    if cust:
        setk("age", _int_or_default(cust.get("age"), DEFAULTS["age"]))
        setk("employment_status", str(cust.get("employment_status", DEFAULTS["employment_status"])).lower())
        inc = _num_or_none(cust.get("income_monthly"))
        setk("income_monthly", inc if inc is not None else DEFAULTS["income_monthly"])
        setk("credit_score", _int_or_default(cust.get("credit_score"), DEFAULTS["credit_score"]))

    if app_:
        amt = _num_or_none(app_.get("amount"))
        if amt is not None:
            setk("amount", amt)

        tm = app_.get("term_months")
        setk("term_months", _int_or_default(tm, st.session_state.get("term_months", DEFAULTS["term_months"])))

        dm = _num_or_none(app_.get("debts_monthly"))
        if dm is not None:
            setk("debts_monthly", dm)

        cv = _num_or_none(app_.get("collateral_value"))
        if cv is not None:
            setk("collateral_value", cv)
        else:
            st.session_state["collateral_value"] = 0.0

        pur = app_.get("purpose")
        if pur is not None and str(pur).strip() != "" and not pd.isna(pur):
            setk("purpose", str(pur).lower())
    else:
        st.session_state["collateral_value"] = 0.0

    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

def safe_index(options, value, default_index=0):
    try:
        return options.index(value)
    except Exception:
        return default_index

# -------------------- Build form --------------------
customer_ids = load_customers() or ["10001"]
st.session_state.setdefault("cust_id", customer_ids[0])

def on_customer_change():
    reset_form_defaults()
    apply_autofill_from_csv(st.session_state["cust_id"], overwrite=True)

# Top controls
top = st.columns([4, 1])
with top[0]:
    st.selectbox("Customer", customer_ids, key="cust_id", on_change=on_customer_change)
with top[1]:
    if st.button("Autofill", use_container_width=True):
        reset_form_defaults()
        apply_autofill_from_csv(st.session_state["cust_id"], overwrite=True)

# First load
if st.session_state.get("FIRST_AUTOFILL_DONE") is not True:
    apply_autofill_from_csv(st.session_state["cust_id"], overwrite=False)
    st.session_state["FIRST_AUTOFILL_DONE"] = True

with st.form("review_form", clear_on_submit=False):
    left, right = st.columns(2)

    with left:
        st.number_input("Age", min_value=18, max_value=75,
                        value=int(st.session_state.get("age", DEFAULTS["age"])),
                        key="age")
        st.selectbox("Employment", EMPLOYMENT_OPTIONS,
                     index=safe_index(EMPLOYMENT_OPTIONS, st.session_state.get("employment_status", DEFAULTS["employment_status"])),
                     key="employment_status")
        st.number_input("Income / mo (LKR)", min_value=1.0, step=1000.0,
                        value=float(st.session_state.get("income_monthly", DEFAULTS["income_monthly"])),
                        key="income_monthly")
        st.number_input("Debts / mo (LKR)", min_value=0.0, step=1000.0,
                        value=float(st.session_state.get("debts_monthly", DEFAULTS["debts_monthly"])),
                        key="debts_monthly")
        st.number_input("Credit score", min_value=300, max_value=850, step=1,
                        value=int(st.session_state.get("credit_score", DEFAULTS["credit_score"])),
                        key="credit_score")

    with right:
        st.number_input("Amount (LKR)", min_value=1.0, step=1000.0,
                        value=float(st.session_state.get("amount", DEFAULTS["amount"])),
                        key="amount")
        st.number_input("Term (months)", min_value=6, max_value=420, step=1,
                        value=int(st.session_state.get("term_months", DEFAULTS["term_months"])),
                        key="term_months")
        st.number_input("Collateral (LKR, optional)", min_value=0.0, step=1000.0,
                        value=float(st.session_state.get("collateral_value", DEFAULTS["collateral_value"])),
                        key="collateral_value")
        st.selectbox("Purpose", PURPOSE_OPTIONS,
                     index=safe_index(PURPOSE_OPTIONS, st.session_state.get("purpose", DEFAULTS["purpose"])),
                     key="purpose")

    cta = st.columns([1, 6])
    with cta[0]:
        submitted = st.form_submit_button("Review", use_container_width=True)
    with cta[1]:
        st.caption("LLM explanation included. Toggle LangGraph in the sidebar.")

# -------------------- Submit handlers --------------------
def _payload():
    payload = {
        "customer_id": str(st.session_state["cust_id"]),
        "age": int(st.session_state["age"]),
        "income_monthly": float(st.session_state["income_monthly"]),
        "debts_monthly": float(st.session_state["debts_monthly"]),
        "amount": float(st.session_state["amount"]),
        "term_months": int(st.session_state["term_months"]),
        "credit_score": int(st.session_state["credit_score"]),
        "employment_status": st.session_state["employment_status"],
        "collateral_value": float(st.session_state["collateral_value"]),
        "purpose": st.session_state["purpose"],
    }
    # Clean any non-JSON numbers
    for k in ["income_monthly", "debts_monthly", "amount"]:
        v = payload.get(k)
        if not math.isfinite(v):
            payload[k] = 0.0
    cv = payload.get("collateral_value")
    if (cv is None) or (not math.isfinite(cv)) or (cv == 0.0):
        payload["collateral_value"] = None
    return payload

def do_request(path: str, payload: dict):
    try:
        res = requests.post(f"{api_base()}{path}", json=payload, timeout=30)
        if not res.ok:
            st.error(f"Request failed: {res.status_code}")
            st.text(res.text)
            return None
        return res.json()
    except Exception as e:
        st.error(f"Request error: {e}")
        return None

def _metric_row(data: dict):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        decision = str(data.get("decision", "—"))
        st.markdown(
            f'<div style="margin-top:-6px;margin-bottom:6px">'
            f'<span class="pill pill-{decision}">{decision}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.metric("Confidence", f"{int(float(data.get('confidence', 0))*100)}%")
    with c3:
        st.metric("DTI", data.get("dti", "—"))
    with c4:
        st.metric("LTV", data.get("ltv", "—"))

def render_decision(data: dict):
    st.success("Decision received")
    _metric_row(data)

    st.caption(
        f"Risk: **{data.get('risk_score', '—')}** · "
        f"KYC: **{data.get('kyc_status', 'unknown')}** · "
        f"AML: **{data.get('aml_hit', '—')}**"
    )

    notes = data.get("reasons") or []
    if notes:
        st.markdown("**Notes**")
        for r in notes:
            st.markdown(f"- {r}")

    hits = data.get("rule_hits") or []
    if hits:
        st.markdown("**Rule trace**")
        df = pd.DataFrame(hits)
        cols = [c for c in ["code", "message", "severity"] if c in df.columns]
        st.dataframe(df[cols] if cols else df, use_container_width=True)

    llm_text = data.get("llm_explanation") or data.get("explanation")
    if llm_text:
        st.markdown('<div class="section-title">🧠 LLM Explanation</div>', unsafe_allow_html=True)
        st.write(llm_text)

    with st.expander("Full response JSON"):
        st.json(data)

if submitted:
    path = "/graph/review" if st.session_state.get("USE_GRAPH", True) else "/review+explain"
    data = do_request(path, _payload())
    if data:
        render_decision(data)

# -------------------- Quick checks --------------------
st.divider()
st.subheader("Quick Checks (KYC / Credit / AML)")
qc1, qc2, qc3 = st.columns(3)
with qc1:
    if st.button("View KYC"):
        r = requests.get(f"{api_base()}/mock/kyc/{st.session_state['cust_id']}", timeout=10)
        st.json(r.json() if r.ok else r.text)
with qc2:
    if st.button("View Credit"):
        r = requests.get(f"{api_base()}/mock/credit/{st.session_state['cust_id']}", timeout=10)
        st.json(r.json() if r.ok else r.text)
with qc3:
    if st.button("View AML"):
        r = requests.get(f"{api_base()}/mock/aml/{st.session_state['cust_id']}", timeout=10)
        st.json(r.json() if r.ok else r.text)

# -------------------- Recent decisions --------------------
st.divider()
st.subheader("Recent Decisions")

try:
    from tzlocal import get_localzone
    _LOCAL_TZINFO = get_localzone()
except Exception:
    _LOCAL_TZINFO = datetime.now().astimezone().tzinfo

payload, items, used_local_fallback = None, [], False

try:
    r = requests.get(f"{api_base()}/decisions?limit=50", timeout=10)
    payload = r.json() if r.ok else None
except Exception as e:
    payload = {"error": str(e)}

def _normalize_items(p):
    if not p:
        return []
    if isinstance(p, dict) and "items" in p and isinstance(p["items"], list):
        return p["items"]
    if isinstance(p, list):
        return p
    return []

items = _normalize_items(payload)

# CSV fallback
if not items:
    try:
        df_local = pd.read_csv("data/decisions_log.csv", engine="python", on_bad_lines="skip")
        if not df_local.empty:
            items = df_local.to_dict(orient="records")
            used_local_fallback = True
    except Exception:
        pass

if items:
    df = pd.DataFrame(items).copy()
    df = df.rename(columns={
        "out_decision": "decision",
        "out_dti": "dti",
        "out_ltv": "ltv",
        "out_risk_score": "risk_score",
        "out_confidence": "confidence",
    })
    if "ts_local" not in df.columns:
        if "ts" in df.columns:
            ts_parsed = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df["ts_local"] = (
                ts_parsed
                .dt.tz_convert(_LOCAL_TZINFO)
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            ).fillna("")
        else:
            df["ts_local"] = ""
    sort_key = None
    if "ts" in df.columns:
        sort_key = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    elif "ts_local" in df.columns:
        sort_key = pd.to_datetime(df["ts_local"], errors="coerce")
    if sort_key is not None:
        df = df.assign(_sort=sort_key).sort_values("_sort").drop(columns="_sort")
    pref = [c for c in ["ts_local","customer_id","decision","dti","ltv","risk_score","confidence","purpose"]
            if c in df.columns]
    rest = [c for c in df.columns if c not in pref]
    st.success(f"Showing {len(df)} recent decisions" + (" (local fallback)" if used_local_fallback else ""))
    st.dataframe(df[pref + rest] if pref else df, use_container_width=True, height=340)
else:
    st.info("No decisions logged yet.")

with st.expander("Raw /decisions payload (debug)"):
    st.write(payload)
