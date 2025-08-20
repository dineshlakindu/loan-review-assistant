# LLM Prompt & Agent Chain (Design Notes)

## Purpose
Explain the loan decision (Approve / Flag / Reject) in **plain language** using a **local LLM** (Ollama). The LLM never decides; it **just explains** the policy outcome with a short paragraph and a confidence hint.

---

## Runtime & Model
- **Engine:** Ollama (local)
- **Default model:** `llama3.2:3b` (override via env `OLLAMA_MODEL`)
- **Endpoint:** `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- **Code paths:** `api/engine/agent.py`, optional graph in `api/engine/agent_graph.py`

### Decoding parameters (stable, non-creative)
```json
{
  "temperature": 0.2,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "num_predict": 220,
  "stop": ["\n- ", "\n* ", "\n• ", "\n1. "]
}
