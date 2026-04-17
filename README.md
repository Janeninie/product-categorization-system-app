# Smart Product Categorization System

ML-powered web application that classifies product images into **beverage** or **snack** categories.

## Tech Stack

- **Frontend:** Next.js 16 (App Router), React, Tailwind CSS, TypeScript, Axios, React Query (TanStack Query)
- **Backend:** FastAPI (Python), PyTorch, safetensors, Pillow
- **Database:** SQLite with SQLAlchemy

## Code Level Architecture

```mermaid
graph TD

    %% ─── ENTRY POINT ───────────────────────────────────────────────
    EP(["⚡ ENTRY POINT\nuvicorn main:app"])

    %% ─── FRONTEND ──────────────────────────────────────────────────
    subgraph FE ["Frontend · Next.js (port 3000)"]
        PAGE["app/page.tsx\n─────────────────\nHome()\nhandleSubmit()\nhandleFeedbackSubmit()\nhandleFileChange()"]
        HOOKS["lib/hooks.ts\n─────────────────\nusePrediction()\nuseHistory()\nuseFeedback()\nuseHealth()"]
        API_LIB["lib/api.ts\n─────────────────\napiClient (axios)\npredictionApi.predict()\npredictionApi.getHistory()\npredictionApi.submitFeedback()\npredictionApi.getHealth()"]
        TYPES["app/api/types.ts\n─────────────────\nPredictionResponse\nHistoryResponse\nFeedbackRequest\nFeedbackResponse\nHealthResponse"]
    end

    %% ─── BACKEND API ────────────────────────────────────────────────
    subgraph BE ["Backend API · FastAPI (port 8000)"]
        MAIN["backend/main.py ⚡\n─────────────────\nlifespan()\nPOST /predict\nGET /history\nGET /healthz\nPOST /feedback\nvalidate_image_format()\nrun_orchestrator_background_task()"]
        SCHEMAS["backend/schemas.py\n─────────────────\nPredictionResponse\nHistoryResponse\nHistoryItem\nFeedbackRequest\nFeedbackResponse\nHealthResponse"]
    end

    %% ─── ML INFERENCE ───────────────────────────────────────────────
    subgraph ML ["ML Inference Layer"]
        ML_MODEL["backend/ml_model.py\n─────────────────\nbuild_model()\nProductClassifier\nSimpleCNN\n_TransferModel\n_build_resnet50()\n_build_mobilenetv3_large()\n_build_convnext_tiny()"]
        QUALITY["backend/quality.py\n─────────────────\nanalyze_quality()\nQualityMetrics\ncalculate_brightness()\ncalculate_blur_var()"]
    end

    %% ─── DRIFT MONITORING ───────────────────────────────────────────
    subgraph DM ["Drift Monitoring Layer"]
        ORCH["backend/orchestrator.py\n─────────────────\nrun_orchestrator_from_db()\ncompute_drift_for_latest_window()\nget_runtime_components()\nload_latest_window_from_db()\nnormalize_class_name()\nin_alert_cooldown()\nbuild_alert_message()"]
        DRIFT_SCRIPT["scripts/compute_drift.py\n─────────────────\nbuild_model()\nbuild_transform()\nload_reference_stats()\nload_reference_embedding_mean()\ninfer_recent_embeddings()\ncompute_embedding_drift()\ncompute_confidence_drift()\ncompute_class_ratio_drift()"]
    end

    %% ─── DATA / ORM ─────────────────────────────────────────────────
    subgraph DB_LAYER ["Data / ORM Layer"]
        DATABASE["backend/database.py\n─────────────────\nPredictionEvent\nHumanFeedback\nDriftEvent\nAlert\ninit_db()\nget_db()\nSessionLocal"]
        SQLITE[("SQLite\nproduct_categorization.db\n─────────────────\nprediction_events\nhuman_feedback\ndrift_events\nalerts\nsystem_state")]
    end

    %% ─── SHARED MONITORING STORE ────────────────────────────────────
    subgraph STORE_LAYER ["Shared Monitoring Store"]
        STORE["src/monitoring/store.py\n─────────────────\ninit_db()\ninsert_drift_event()\ninsert_alert()\nDB_PATH"]
    end

    %% ─── EDGES · FRONTEND ───────────────────────────────────────────
    PAGE -->|"uses hooks"| HOOKS
    HOOKS -->|"calls predictionApi.*"| API_LIB
    API_LIB -->|"imports types"| TYPES
    HOOKS -->|"imports types"| TYPES

    %% ─── EDGES · FRONTEND → BACKEND HTTP ────────────────────────────
    API_LIB -->|"POST /predict multipart"| MAIN
    API_LIB -->|"GET /history"| MAIN
    API_LIB -->|"POST /feedback"| MAIN
    API_LIB -->|"GET /healthz"| MAIN

    %% ─── EDGES · BACKEND STARTUP ────────────────────────────────────
    EP -->|"starts"| MAIN
    MAIN -->|"lifespan: init_db()"| DATABASE
    MAIN -->|"lifespan: build_model()"| ML_MODEL

    %% ─── EDGES · PREDICT ENDPOINT ───────────────────────────────────
    MAIN -->|"analyze_quality()"| QUALITY
    MAIN -->|"classifier.predict()"| ML_MODEL
    MAIN -->|"PredictionEvent.save"| DATABASE
    MAIN -->|"validates with"| SCHEMAS
    MAIN -->|"BackgroundTasks.add_task"| ORCH

    %% ─── EDGES · HISTORY / FEEDBACK ─────────────────────────────────
    MAIN -->|"query PredictionEvent"| DATABASE
    MAIN -->|"save HumanFeedback"| DATABASE

    %% ─── EDGES · ORCHESTRATOR ───────────────────────────────────────
    ORCH -->|"init_db()"| STORE
    ORCH -->|"insert_drift_event()"| STORE
    ORCH -->|"insert_alert()"| STORE
    ORCH -->|"load_reference_stats()\nbuild_model() / build_transform()\ninfer_recent_embeddings()\ncompute_*_drift()"| DRIFT_SCRIPT
    ORCH -->|"raw sqlite reads"| SQLITE

    %% ─── EDGES · DATA LAYER ─────────────────────────────────────────
    DATABASE -->|"SQLAlchemy engine"| SQLITE
    STORE    -->|"sqlite3 direct"| SQLITE
```

### Module Summary

| Module                         | Responsibility                                                                                                                                          | Depends On                                                                  | Exposed Functions / Classes                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **`app/page.tsx`** ⚡          | Root UI page — file upload, prediction display, feedback form, history table                                                                            | `lib/hooks.ts`, `app/api/types.ts`                                          | `Home()`                                                                                                                                        |
| **`lib/hooks.ts`**             | React Query hooks bridging UI state to API calls                                                                                                        | `lib/api.ts`, `app/api/types.ts`                                            | `usePrediction()`, `useHistory()`, `useFeedback()`, `useHealth()`                                                                               |
| **`lib/api.ts`**               | Axios client configured against FastAPI base URL; wraps all four endpoints                                                                              | `app/api/types.ts`                                                          | `apiClient`, `predictionApi.predict()`, `predictionApi.getHistory()`, `predictionApi.submitFeedback()`, `predictionApi.getHealth()`             |
| **`app/api/types.ts`**         | Shared TypeScript interfaces for all API request/response bodies                                                                                        | —                                                                           | `PredictionResponse`, `HistoryResponse`, `HistoryItem`, `FeedbackRequest`, `FeedbackResponse`, `HealthResponse`                                 |
| **`backend/main.py`** ⚡       | FastAPI app — HTTP routing, image decoding, model invocation, DB persistence, background drift trigger                                                  | `database.py`, `ml_model.py`, `quality.py`, `orchestrator.py`, `schemas.py` | `POST /predict`, `GET /history`, `GET /healthz`, `POST /feedback`                                                                               |
| **`backend/schemas.py`**       | Pydantic request/response models for automatic FastAPI validation and serialisation                                                                     | —                                                                           | `PredictionResponse`, `HistoryResponse`, `FeedbackRequest`, `FeedbackResponse`, `HealthResponse`                                                |
| **`backend/ml_model.py`**      | Model architecture definitions and `build_model()` factory (EfficientNet-B0, SimpleCNN, ResNet-50, MobileNetV3, ConvNeXt variants)                      | PyTorch, torchvision                                                        | `build_model()`, `ProductClassifier`, `SimpleCNN`, `_TransferModel`                                                                             |
| **`backend/quality.py`**       | Analyses PIL images for brightness, blur variance, and resolution; emits quality warnings                                                               | Pillow, OpenCV, NumPy                                                       | `analyze_quality()`, `QualityMetrics`                                                                                                           |
| **`backend/database.py`**      | SQLAlchemy ORM models, engine, session factory, and `init_db()` migration helper                                                                        | SQLAlchemy, SQLite                                                          | `PredictionEvent`, `HumanFeedback`, `DriftEvent`, `Alert`, `init_db()`, `get_db()`, `SessionLocal`                                              |
| **`backend/orchestrator.py`**  | Drift-check coordinator — reads new predictions from DB, delegates to `compute_drift.py`, writes drift events/alerts. Runs as a FastAPI background task | `src/monitoring/store.py`, `scripts/compute_drift.py`, SQLite               | `run_orchestrator_from_db()`                                                                                                                    |
| **`scripts/compute_drift.py`** | Stateless drift math library — decodes base64 images, extracts channel-stat embeddings, computes embedding / confidence / class-ratio drift scores      | NumPy, Pillow                                                               | `load_reference_stats()`, `infer_recent_embeddings()`, `compute_embedding_drift()`, `compute_confidence_drift()`, `compute_class_ratio_drift()` |
| **`src/monitoring/store.py`**  | Raw `sqlite3` helpers to create and write `drift_events` and `alerts` tables                                                                            | SQLite (stdlib)                                                             | `init_db()`, `insert_drift_event()`, `insert_alert()`, `DB_PATH`                                                                                |
| **SQLite DB**                  | Single-file relational store shared by both SQLAlchemy ORM and the orchestrator's direct `sqlite3` connection                                           | —                                                                           | Tables: `prediction_events`, `human_feedback`, `drift_events`, `alerts`, `system_state`                                                         |

> **Legend**: ⚡ = system entry point · internal-only functions are not listed in the table

## Frontend Architecture

The frontend uses a modern data-fetching architecture:

- **Axios** - HTTP client for API communication with interceptors and error handling
- **React Query (TanStack Query)** - Server state management with caching, automatic refetching, and optimistic updates

### Key Files

```
frontend/
├── app/
│   ├── page.tsx          # Main page component
│   ├── layout.tsx        # Root layout with providers
│   ├── providers.tsx     # React Query provider setup
│   └── api/types.ts      # TypeScript interfaces for API responses
├── lib/
│   ├── api.ts            # Axios API client and endpoints
│   └── hooks.ts          # React Query hooks (usePrediction, useHistory, useFeedback)
```

### React Query Hooks

| Hook                        | Purpose                         |
| --------------------------- | ------------------------------- |
| `usePrediction()`           | Upload image and get prediction |
| `useHistory(limit, offset)` | Fetch prediction history        |
| `useFeedback()`             | Submit human feedback           |
| `useHealth()`               | Check backend health status     |

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and proxies API requests to the backend at `http://localhost:8000`.

## API Endpoints

| Endpoint    | Method | Description                                          |
| ----------- | ------ | ---------------------------------------------------- |
| `/predict`  | POST   | Upload an image (JPG/PNG) for classification         |
| `/history`  | GET    | Get prediction history (`?limit=20&offset=0`)        |
| `/healthz`  | GET    | Health check                                         |
| `/feedback` | POST   | Submit human feedback for low-confidence predictions |

## Prediction Response

```json
{
  "predicted_class": "beverage",
  "confidence": 0.95,
  "latency_ms": 125.5,
  "low_confidence_flag": false,
  "brightness": 128.3,
  "blur_var": 45.2,
  "width": 224,
  "height": 224,
  "quality_warnings": [],
  "prediction_id": 1
}
```

## Database Schema

- **prediction_events** - Records all predictions with quality metrics
- **human_feedback** - Stores corrections for low-confidence predictions
- **drift_events** - Data drift monitoring records
- **alerts** - System alerts for administrators

## Reset Database (Development)

If you want to clear all prediction history and feedback data, reset the SQLite file.

1. Stop the backend server.
2. Remove the database file.
3. Start the backend again (tables are created automatically at startup).

```bash
rm -f backend/product_categorization.db
```

Then run backend again:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
```

Notes:

- This permanently deletes all local data in all tables.
- This is intended for local development only.

## Model

The system uses EfficientNet-B0 for product classification with 2 classes: `beverage` and `snack`.
