# Auto-CRO: Project Context & Architecture

**Role:** You are a Senior ML Engineer and Python Developer.
**Goal:** Build a Contextual Thompson Sampling system that optimizes UI elements based on visual features extracted via Vision-Language Models (VLM).

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| ML Core | PyTorch | `nn.Module`, GPU via `non_blocking=True` |
| VLM Provider | OpenRouter | dev: `gemini-2.0-flash-001`, prod: `claude-3.5-sonnet` |
| Async HTTP | httpx | `AsyncClient` only, never `requests` |
| Validation | Pydantic v2 | `model_validate_json` for VLM output |
| Backend | FastAPI + Uvicorn | async routes, dependency injection |
| Database | Supabase | PostgreSQL, bandit state + logs |
| MLOps | WandB | experiment tracking, uncertainty logging |
| Package manager | uv | `pyproject.toml` |
| Linter | Ruff + mypy strict | run before every commit |

## Critical Contracts
- **`contracts.py`** — единственный источник типов между компонентами.
  - `FEATURE_DIM = 4`, `N_ARMS = 3`
  - `DecisionRequest`, `DecisionResponse`, `FeedbackPayload`, `BanditState`, `StepLog`
- **Изменения в `contracts.py` вносит только координатор (человек).**

## Data Flow
```
image_bytes → VLMFeatureExtractor.extract() → UIFeatures (Pydantic)
  → context: ContextTensor [1, FEATURE_DIM]
  → ThompsonBandit.sample(context) → ArmIndex
  → Frontend показывает вариант
  → FeedbackPayload(arm_index, reward) → ThompsonBandit.update()
  → BanditState → Supabase (sync_with_db)
  → StepLog → WandB
```

---

## Agent Boundaries

### 🧠 Agent ML (`ml_core/`)
**Owns:** `ml_core/**`, `main_pipeline.py`
**Forbidden:** `backend/**`, `tests/**`, `contracts.py`
**Current tasks:**
- Fix bug in `extractor.py`: move `response.json()` inside `async with` block
- Implement `main_pipeline.py` (3 TODOs: extract → tensor → sample)
- Implement `ml_core/storage.py` (Supabase serialization of tensors)
- Implement `ml_core/mlops.py` (WandB logging via `StepLog`)
- Implement `ThompsonBandit.sync_with_db()` using `storage.py`

### ⚙️ Agent Backend (`backend/`)
**Owns:** `backend/**`
**Forbidden:** `ml_core/**`, `contracts.py`
**Current tasks:**
- Implement `backend/app.py` (FastAPI app factory)
- Implement `backend/api/deps.py` (bandit singleton via `@lru_cache`)
- Implement `backend/api/routes.py` (POST /decide, POST /feedback)
- Implement `backend/schemas/requests.py` and `responses.py` (replace TypedDict in contracts.py)

### 🧪 Agent Tests (`tests/`)
**Owns:** `tests/**`
**Forbidden:** modifying source files directly; only read them
**Current tasks:**
- `tests/test_thompson.py`: unit tests for `update()` and `sample()`
- `tests/test_extractor.py`: mock httpx to test `extract()` without real API call
- `tests/test_api.py`: FastAPI `TestClient` integration tests for /decide and /feedback

### 📊 Agent MLOps (`ml_core/mlops.py`, `ml_core/storage.py`, `mcp_supabase_server.py`)
**Owns:** `ml_core/mlops.py`, `ml_core/storage.py`, `mcp_supabase_server.py`, `docs/**`
**Forbidden:** `ml_core/bandits/**`, `ml_core/vlm/**`, `backend/**`
**Current tasks:**
- Implement `ml_core/mlops.py`: WandB init + `log_step(StepLog)` function
- Implement `ml_core/storage.py`: `save_state()` / `load_state()` with Supabase + base64 tensors
- Implement `check_supabase` tool in `mcp_supabase_server.py`

---

## Strict Rules (apply to ALL agents)
1. All public functions/methods MUST have Type Hints (`mypy --strict`)
2. All classes MUST have Google-style docstrings
3. No `print()` — use `logging` module
4. No `requests` — use `httpx.AsyncClient` for all HTTP
5. All JSON structures from external APIs validated via Pydantic
6. Import types from `contracts.py`, never redefine them locally