# Loan Review Assistant (IS4007 Demo)

A small single-agent system that reviews synthetic loan applications using **rules + local LLM explanation**, with a **FastAPI** backend and **Streamlit** UI. Simulated external systems: **KYC / Credit / AML**.

---

## Quick Start (Windows / PowerShell)

### 0) Requirements
- Python 3.11+ (recommended 3.12), Git, VS Code
- [Ollama](https://ollama.com) installed and running (for local LLM)

### 1) Create venv & install deps
```powershell
cd <your>\loan_project_updated_fixed
py -3.12 -m venv .venv
# If PowerShell blocks activation, run once:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

