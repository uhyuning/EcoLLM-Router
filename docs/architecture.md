# EcoLLM Router — Architecture

## Request Flow

```
User Prompt
    │
    ▼
POST /chat/
    │
    ├─► Classifier (classifier.py)
    │       ├─ ML model (if artifact exists)
    │       └─ Heuristic fallback (if not trained yet)
    │           → complexity_score ∈ [0, 1]
    │
    ├─► Router (router.py)
    │       score < threshold  →  Gemini Flash  (cheap, fast)
    │       score ≥ threshold  →  Gemini Pro    (smart, costly)
    │
    └─► LLM Client (llm_client.py)
            Exponential back-off retry (max 3 attempts)
            → answer, token counts, latency_ms
    │
    ▼
ChatResponse  (answer + cost + latency + model_used)
```

## Key Modules

| Path | Role |
|---|---|
| `app/core/config.py` | All tuneable parameters (models, costs, thresholds) |
| `app/services/classifier.py` | Complexity scoring — ML or heuristic |
| `app/services/router.py` | Model selection logic |
| `app/services/llm_client.py` | Async Gemini API wrapper with retry |
| `app/api/routes/chat.py` | Main POST endpoint |
| `app/api/routes/metrics.py` | In-process call statistics |
| `classifier/src/features.py` | Feature extraction (shared by train & serve) |
| `classifier/src/train.py` | Train & save the GBM classifier |
| `benchmarks/scripts/` | Baseline vs routed cost/latency comparison |

## Routing Decision

```
complexity_score = classifier.predict(prompt)

if score >= COMPLEXITY_THRESHOLD (default 0.5):
    → gemini-2.5-pro
else:
    → gemini-2.0-flash
```

## Cost Model (per 1M tokens, approximate)

| Model | Input | Output |
|---|---|---|
| gemini-2.0-flash | $0.075 | $0.30 |
| gemini-2.5-pro | $1.25 | $10.00 |
