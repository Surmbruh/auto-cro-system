---
name: Agent Backend
description: Implements FastAPI backend - routes, dependency injection, Pydantic schemas for API layer.
---

# Agent Backend — Инструкция

## Твоя зона ответственности
- `backend/app.py` — фабрика FastAPI приложения
- `backend/api/routes.py` — HTTP-эндпоинты
- `backend/api/deps.py` — dependency injection
- `backend/schemas/requests.py` — входные Pydantic-схемы
- `backend/schemas/responses.py` — выходные Pydantic-схемы

## Запрещено
- Трогать `ml_core/**`, `tests/**`
- Изменять `contracts.py` (только читать для справки)
- Добавлять бизнес-логику ML в роуты (только вызовы через deps)

## Архитектурный паттерн: Router → Service → ML Core
```
POST /decide
  ↓ routes.py (валидация HTTP)
  ↓ deps.py (получить bandit singleton)
  ↓ main_pipeline.run_optimization_step() (ML логика)
  ↓ routes.py (сформировать HTTP ответ)
```

## Задачи в порядке приоритета

### 1. `backend/api/deps.py` — Singleton бандита
- `ThompsonBandit` должен жить как синглтон между запросами
- Используй `@lru_cache` или `lifespan` FastAPI
- Смотри скелет файла

### 2. `backend/schemas/requests.py`
```python
class DecideRequest(BaseModel):
    # image передаётся как multipart/form-data, НЕ base64 в JSON
    pass  # см. скелет

class FeedbackRequest(BaseModel):
    arm_index: int
    reward: float  # 0.0 или 1.0
```

### 3. `backend/schemas/responses.py`
```python
class DecideResponse(BaseModel):
    arm_index: int
    confidence: float
```

### 4. `backend/api/routes.py` — Эндпоинты
- `POST /decide` — принять image bytes, вернуть `DecideResponse`
- `POST /feedback` — принять `FeedbackRequest`, вернуть `{"status": "ok"}`
- `GET /health` — healthcheck

### 5. `backend/app.py` — Приложение
- Использовать `lifespan` для инициализации бандита при старте
- Подключить router из routes.py
- Добавить CORS если нужен frontend

## Соглашения
- Все эндпоинты асинхронные (`async def`)
- `UploadFile` для приёма изображений (multipart)
- Ошибки оборачивать в `HTTPException` с понятными статусами
- Импортировать типы из `contracts.py`, схемы — из `backend/schemas/`
