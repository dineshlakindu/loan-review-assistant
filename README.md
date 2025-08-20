![CI](https://github.com/dineshlakindu/loan-review-assistant/actions/workflows/ci.yml/badge.svg?branch=main)

# Loan Review Assistant (IS4007 Demo)

A small single-agent system that reviews **synthetic loan applications** using **rules + local LLM explanation**, with a **FastAPI** backend and **Streamlit** UI.  
Simulated external systems (mock APIs): **KYC / Credit / AML**. A LangGraph agent route is included and safely disabled in CI if LangGraph isn’t installed.

---

## Features
- **FastAPI** service with typed schemas and a JSON-safe `/decisions` log
- **Streamlit** UI: form entry, CSV autofill, explanations, recent decisions table
- **Mock services**: `/mock/kyc/{id}`, `/mock/credit/{id}`, `/mock/aml/{id}`
- **Agent route** (optional): `/graph/review` via LangGraph
- **Tests + CI**: `pytest` with GitHub Actions badge above

---

## Quickstart

### Requirements
- Python **3.11+** (tested on 3.12), Git, VS Code
- (Optional) [Ollama](https://ollama.com) running locally for LLM explanations

### 1) Clone & create venv

#### Windows (PowerShell)
```powershell
git clone https://github.com/dineshlakindu/loan-review-assistant.git
cd loan-review-assistant

py -3.12 -m venv .venv
# If activation is blocked the first time:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
