# API Endpoints (FastAPI)
Base URL: `http://127.0.0.1:8000` • Docs: `/docs`

## Core
- `POST /review`
- `POST /review+explain`
- `POST /review/autofill`
- `POST /review+explain/autofill`
- `GET /decisions?limit=50`

## Customers / Mock
- `GET /customers`
- `GET /customers/{id}`
- `GET /mock/kyc/{id}`
- `GET /mock/aml/{id}`
- `GET /mock/credit/{id}`

## Agent
- `POST /graph/review`
