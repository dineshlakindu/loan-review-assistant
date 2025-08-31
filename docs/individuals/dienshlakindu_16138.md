# Dinesh Lakindu — 16138
**Component owned:** LLM prompt & LangGraph agent

## 1. Summary (2–3 lines)
Implemented the explanation LLM flow and a LangGraph-based agent path for loan review. The agent calls mock KYC/AML, applies rules, and returns a structured decision with human-readable reasons.

## 2. Responsibilities
- Designed the **structured explanation** prompt (`llm_explain_structured`) used by `/review+explain`.
- Added **LangGraph agent** route (`/graph/review`) and orchestration in `api/engine/agent_graph.py`.
- Aligned outputs (decision, risk_score, reasons, confidence, rule_hits) with UI needs.

## 3. Design decisions
- **Structured output**: forced model to return JSON fields so UI can render reasons & confidence reliably.
- **Fallback logic**: if the structured helper fails, fall back to a simple `llm_explain`.
- **Safety**: capped confidence and preserved rule-driven decision as source of truth.

## 4. Files touched (main)
- `api/engine/agent.py` — explanation helpers (`llm_explain_structured`, fallback).
- `api/engine/agent_graph.py` — LangGraph flow + runner.
- `api/main.py` — endpoint wiring for `/review+explain` and `/graph/review`.

## 5. How to run my part
1. Start API: `uvicorn api.main:app --reload`
2. (Optional) Run local LLM (Ollama) as per README.
3. Call: `POST /review+explain` or `POST /graph/review` with a valid loan payload.


## 6. Testing / validation
- Verified the response includes: `decision`, `risk_score`, `reasons`, `rule_hits`, `confidence`.
- Confirmed CI passes (`pytest -q`), and API returns JSON-safe values.

## 7. Challenges & lessons
- Getting consistent JSON from the LLM → used few-shot + JSON schema hints.
- Kept the **rules engine authoritative**; LLM only explains/augments.

## 8. Next steps
- Add few-shot examples for edge cases (e.g., borderline DTI).
- Improve error handling if the LLM service is down.
