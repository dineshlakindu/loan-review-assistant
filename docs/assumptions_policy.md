# Loan Review Assistant — Assumptions & Policy (Short)

## Purpose
Fixes the data schema, mock API contracts, and the decision rules used by the system.

## Data (CSV files in /data)
- customers.csv (used for autofill)
  - REQUIRED: customer_id, age (18–75), employment_status (employed|self_employed|contract|student|unemployed),
    income_monthly (>0), debts_monthly (>=0), credit_score (300–850)
  - OPTIONAL: collateral_value (>0), purpose (home|vehicle|education|personal|business)
- kyc.csv: customer_id, kyc_status (pass|fail|pending)
- aml.csv: customer_id, watchlist_hit (true/false)
- decisions_log.csv (append-only audit written by API)
  - ts, customer_id, inputs..., out_decision, out_dti, out_ltv, out_risk_score, out_reasons, out_confidence, out_rule_hits, kyc_status, aml_hit

## Key API Endpoints (FastAPI)
- POST /review — rules only
- POST /review+explain — rules + LLM paragraph
- POST /review/autofill — merge PartialApplication + customers.csv → rules
- POST /review+explain/autofill — same but with LLM
- GET /customers, GET /customers/{id}
- GET /decisions
- GET /health, GET /config
- Agent demo: POST /graph/review
- Mock systems: GET /mock/kyc/{id}, /mock/credit/{id}, /mock/aml/{id}

## Input Schemas
LoanApplication (strict):
- customer_id, age, employment_status, income_monthly, debts_monthly,
  amount (>0), term_months (>0), credit_score, collateral_value (optional), purpose

PartialApplication (for /autofill routes):
- customer_id + (amount, term_months, purpose, collateral_value?, debts_monthly?)
- Server merges missing fields from customers.csv; request values override CSV.

## Derived Metrics
- DTI = round(debts_monthly / income_monthly, 2)  (0 if income = 0)
- LTV = round(amount / collateral_value, 2) if collateral provided, else None

## Decision Rules (Deterministic)
Immediate negatives:
- credit_score < 500  → reason “Very low credit score (<500).”
- income_monthly < 50,000 → reason “Low income (<50,000).” (flag severity)
- employment_status == unemployed → reason “Unemployed status.”
- DTI ≥ 0.60 → reason “High DTI (≥0.6).”
- LTV > 1.20 (if collateral) → reason “High LTV (>1.2).” (flag)

Label when negatives exist:
- If any reject-severity reason (credit <500, DTI ≥0.6, unemployed) → REJECT
- Else → FLAG

Positives:
- credit_score ≥ 700, DTI ≤ 0.40, LTV ≤ 0.80 or no collateral, income ≥ 100,000

Approval:
- If risk ≤ 65 AND at least 2 positives → APPROVE
- Else → FLAG (“Borderline; needs manual review.”)

## Risk Score (0–100; higher = riskier)
Start 50; add:
- DTI: +30 (≥1.0) | +18 (≥0.6) | +10 (≥0.4) | +5 (≥0.3)
- LTV: +25 (>1.2) | +15 (>1.0) | +8 (>0.8) | +5 (no collateral)
- Credit: +30 (<550) | +20 (<600) | +12 (<650) | +6 (<700)
- Employment: +20 (unemployed|student) | +10 (contract)
- Age: +10 (<21)
- Amount vs income: +10 (amount > income_monthly * 20)
- Purpose: +5 (personal)
Clamp to [0,100].

## LLM Explanation
- Local Ollama (default model: llama3.2:3b). Returns 1 short paragraph and confidence.
- If LLM is down → fallback deterministic paragraph.

## Errors & Safety
- /autofill 404 if customer_id not found.
- Validation on ranges (age, credit_score, amounts).
- /decisions reader is safe for empty/corrupt CSV.
