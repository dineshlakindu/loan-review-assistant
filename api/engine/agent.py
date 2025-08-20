# api/engine/agent.py
# Natural-language LLM explainer for loan decisions via Ollama

from __future__ import annotations

import os
import re
import json
import requests
from typing import Dict, Any, List, Tuple

# ---- Ollama config (override via environment) ----
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# Examples: "llama3.2:3b", "mistral", fall back to tiny if needed
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Keep outputs steady and discourage lists
GENERATION_OPTIONS: Dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 220,
    # Block common bullet/numbered list starts
    "stop": ["\n- ", "\n* ", "\n• ", "\n1. ", "\n1) ", "\n2. ", "\n2) "],
}

# ---------- Internal helpers ----------

def _check_ollama_alive() -> None:
    """Raise if Ollama is not reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}. Details: {e}")

def _format_fallback(review_result: Dict[str, Any]) -> str:
    """Deterministic paragraph if LLM is unavailable."""
    d = review_result or {}
    reasons: List[str] = d.get("reasons") or []
    score = d.get("score") or d.get("credit_score") or "N/A"
    kyc = d.get("kyc_status") or d.get("kyc") or "N/A"
    dti = d.get("dti", "N/A")
    ltv = d.get("ltv", "N/A")
    return (
        f"The current decision is {d.get('decision','N/A')}. "
        f"The credit score is {score}, the debt-to-income ratio is {dti}, and the loan-to-value is {ltv}. "
        f"KYC/AML status is {kyc}. "
        f"Key reasons include: {', '.join(reasons) if reasons else 'not specified'}. "
        f"This summary is shown because the local LLM is unavailable."
    )

def _postprocess_to_one_paragraph(text: str) -> str:
    text = re.sub(r'(?mi)^\s*(decision|result|summary)\s*:\s*', '', text)
    text = re.sub(r'(?m)^\s*[-*•]\s*', '', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    text = ' '.join(text.split())
    return text

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _derive_confidence(review_result: Dict[str, Any]) -> float:
    """
    If caller didn't compute confidence, produce a reasonable heuristic:
    - prefer provided 'confidence' (0..1 or 0..100)
    - otherwise, use risk_score and rule severities if available
    """
    # prefer provided
    if "confidence" in review_result and review_result["confidence"] is not None:
        c = float(review_result["confidence"])
        # accept 0..1 or 0..100
        return _clamp(c if c <= 1.0 else c / 100.0)

    # heuristic from risk_score and rule severities
    risk = float(review_result.get("risk_score", 60) or 60.0)  # typical mid
    base = 0.5 + (risk - 50.0) / 100.0  # ~0.4..0.6 around 50
    rule_hits = review_result.get("rule_hits") or []
    # boost for infos, penalize warns/errors
    for rh in rule_hits:
        sev = str(rh.get("severity", "")).lower()
        if sev == "info":
            base += 0.03
        elif sev == "warn":
            base -= 0.05
        elif sev == "error":
            base -= 0.12
    return _clamp(base, 0.25, 0.95)

# ---------- Public API (v1: string only, kept for backward-compat) ----------

def llm_explain(review_result: Dict[str, Any]) -> str:
    """
    Explain the loan decision in one short paragraph (4–6 sentences), in ENGLISH only.
    No headings, no bullet points, no numbered lists, no JSON.
    """
    prompt = (
        "Write ONE short paragraph (4–6 sentences) in ENGLISH ONLY explaining why this loan "
        "was approved, rejected, or flagged. "
        "Weave the DTI, LTV, credit score, and KYC/AML status naturally into your explanation. "
        "Do NOT use bullet points, numbers, lists, tables, or JSON. "
        "Do NOT repeat the prompt or the input. "
        "Do NOT invent any numbers—only use the values given.\n\n"
        f"Decision JSON:\n{json.dumps(review_result, ensure_ascii=False, indent=2)}"
    )

    try:
        _check_ollama_alive()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional loan officer. "
                            "Your job is to produce clear, natural ENGLISH summaries "
                            "as a single paragraph (4–6 sentences). "
                            "Never output bullet points, lists, JSON, tables, or headings."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": GENERATION_OPTIONS,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "").strip()
        return _postprocess_to_one_paragraph(text) or "LLM returned empty text."
    except Exception:
        return _format_fallback(review_result)

# ---------- Public API (v2: structured explanation + confidence) ----------

def llm_explain_structured(review_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: {"explanation": str, "confidence": float (0..1)}
    - Requests JSON from the model; falls back gracefully.
    - If model returns plain text, we still fill both fields.
    """
    system = (
        "You are a concise, professional loan officer. "
        "Explain decisions in one short paragraph (4–6 sentences). "
        "No lists, no headings. Output JSON only."
    )
    user = {
        "instruction": (
            "Explain the decision briefly and output pure JSON with keys "
            "`explanation` (string) and `confidence` (0..1)."
        ),
        "review_result": review_result,
        "constraints": [
            "Use only values given in review_result (no invented numbers).",
            "One short paragraph in `explanation`.",
            "Do not include markdown, lists, or extra keys."
        ],
        "schema": {"explanation": "string", "confidence": "number (0..1)"}
    }

    # default fallback
    fallback_text = _format_fallback(review_result)
    fallback_conf = _derive_confidence(review_result)

    try:
        _check_ollama_alive()
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                "stream": False,
                "options": GENERATION_OPTIONS,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()

        # try JSON first
        try:
            data = json.loads(content)
            expl = str(data.get("explanation", "")).strip()
            conf = data.get("confidence", None)
            if conf is not None:
                conf = float(conf)
                conf = conf if conf <= 1.0 else conf / 100.0
            else:
                conf = fallback_conf
            if not expl:
                # if explanation is empty, try postprocessing plain text
                expl = _postprocess_to_one_paragraph(content)
            return {"explanation": expl or fallback_text, "confidence": _clamp(conf)}
        except Exception:
            # model returned plain text
            return {
                "explanation": _postprocess_to_one_paragraph(content) or fallback_text,
                "confidence": fallback_conf,
            }
    except Exception:
        # Ollama down
        return {"explanation": fallback_text, "confidence": fallback_conf}
