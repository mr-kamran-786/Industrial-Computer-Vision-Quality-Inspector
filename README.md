# Industrial Computer Vision Quality Inspector

AI-powered PCB defect detection for manufacturing quality control. A YOLO model
detects defects; an **auditable decision policy** turns detections into a
PASS/FAIL verdict with full **traceability**, an **operator feedback loop**, and
**quality analytics** (yield, DPPM, escape rate, false-call rate).

> Detects six PCB defect classes: `missing_hole`, `mouse_bite`, `open_circuit`,
> `short`, `spur`, `spurious_copper`.

---

## Why this is built for a factory, not a demo

| Capability | What it gives the quality team |
|---|---|
| **Per-class thresholds + severity policy** | A `short` (functional killer) is treated differently from a cosmetic `spur`. Verdicts are explainable — every decision records *why*. |
| **Traceability** | Each inspection links to a serial number, batch/lot, line, station, operator and shift — the minimum for ISO 9001 / IATF 16949 audit. |
| **Model provenance** | Every record stores the model version + thresholds used, so any past verdict can be reproduced and defended. |
| **Operator feedback loop** | Inspectors confirm/override the AI verdict. This yields the metrics that actually matter: **escape rate** and **false-call rate**. |
| **Quality analytics** | Yield %, DPPM, defect Pareto, severity mix, agreement rate. |
| **Production hardening** | API-key auth, locked CORS, upload validation, structured logging, health/readiness probes, versioned DB migrations, test suite. |

---

## Architecture

```
frontend/ (Next.js)  ──HTTP──▶  backend/app (FastAPI)
                                   ├─ api/routes.py        REST endpoints
                                   ├─ core/config.py       env-driven settings
                                   ├─ core/policy.py       PASS/FAIL decision engine
                                   ├─ core/security.py     API-key auth
                                   ├─ services/            YOLO inference
                                   └─ db/                  models · crud · analytics
                                        │
                                        ▼
                                   SQLite (dev) / PostgreSQL (prod)
                                   + Alembic migrations
ml/  training · evaluation · inference scripts
```

The **policy engine** ([`core/policy.py`](backend/app/core/policy.py)) is the heart
of the system and is deliberately pure/side-effect-free so it can be unit-tested
and audited. Inference runs at a low confidence *floor*; the policy then applies
the stricter per-class thresholds and severity rules.

---

## Setup

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt -r requirements-dev.txt
```

> **Torch**: `requirements.txt` pins CUDA builds. For CPU-only or a different
> CUDA version, install `torch`/`torchvision` from
> <https://pytorch.org/get-started/locally/> first, then the rest.

Copy `.env.example` to `.env` and adjust. Place trained weights in `ml/models/`
(`pcb_defect_yolo_v1.pt`, `pcb_defect_yolo_medium_v1.pt`).

### Database

```bash
alembic upgrade head          # create / migrate schema
```

SQLite is the default. For PostgreSQL, set
`ICVQI_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/icvqi` and re-run
`alembic upgrade head`.

### Run

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- API docs: <http://127.0.0.1:8000/docs>
- Health: `/api/health` · Readiness: `/api/ready`

Frontend:

```bash
cd frontend && npm install && npm run dev
```

---

## API

| Method | Path | Auth | Purpose |
|---|---|:---:|---|
| GET | `/api/health` | — | Liveness |
| GET | `/api/ready` | — | Readiness (model available?) |
| GET | `/api/model/info` | — | Model registry, classes, thresholds, severities |
| POST | `/api/predict/image` | ✅ | Inspect an image (+ traceability form fields) |
| GET | `/api/inspections` | — | Recent inspections (filter by line/status) |
| GET | `/api/inspections/{id}` | — | Full inspection detail |
| POST | `/api/inspections/{id}/review` | ✅ | Record operator ground-truth verdict |
| GET | `/api/analytics/summary` | — | Yield, DPPM, escape/false-call rates, Pareto |

`POST /predict/image` accepts multipart form fields: `file`, `model_key`,
`serial_number`, `batch_lot`, `production_line`, `station_id`, `operator_id`,
`shift`.

**Auth**: when `ICVQI_API_KEYS` is set, protected endpoints require the
`X-API-Key` header. If it is empty, auth is disabled (dev only) and the service
logs a warning on startup.

---

## Quality metrics explained

- **Yield %** — share of units that PASS.
- **DPPM** — defective parts per million (`fail / total × 1e6`).
- **Escape rate** — inspections the AI passed but an operator failed (defects
  that *escaped* the gate). The single most important safety metric.
- **False-call rate** — inspections the AI failed but an operator passed (good
  parts wrongly rejected → scrap cost + operator distrust).
- **Agreement rate** — how often the AI verdict matched the human verdict.

Escape/false-call/agreement are only populated once inspections are reviewed via
the feedback endpoint.

---

## Testing

```bash
pytest
```

Covers the policy engine and the full API surface (prediction, traceability,
review loop, analytics, auth enforcement, upload validation). Inference is
stubbed so tests run without GPU or model weights.

---

## Configuration reference

All settings are environment variables prefixed `ICVQI_` (see `.env.example`):
database URL, CORS origins, API keys, upload size limit, default model, image
size, global confidence threshold, fail-triggering severities, and minor-defect
tolerance. Per-class thresholds and class→severity mapping live in
[`core/config.py`](backend/app/core/config.py).

---

## Roadmap

- **Done — backend:** decision policy, traceability, feedback loop, security,
  analytics, migrations, tests.
- **Done — ML pipeline:** defect-aware training config, per-class threshold
  tuning wired into the live policy, evaluation reports (md + JSON), ONNX export
  with graph validation, and a JSON model registry. See [`ml/README.md`](ml/README.md).
- **Done — frontend:** SPC defect-rate control chart (mean + ±2σ band),
  yield/DPPM tiles, feedback-quality KPIs (escape / false-call / agreement),
  traceability capture + display, severity mix, and an operator review UI — all
  in `frontend/` (Next.js 16 + React 19).
- **Next:** camera/RTSP real-time ingestion; grow the dataset 10–50× to reach
  >99% recall on critical defects; drift monitoring + retraining from operator
  feedback.
```
