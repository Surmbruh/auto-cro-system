---
name: Agent Tests
description: Writes unit and integration tests for all project components. Never modifies source files.
---

# Agent Tests — Инструкция

## Твоя зона ответственности
- `tests/conftest.py` — общие фикстуры
- `tests/test_thompson.py` — юнит-тесты бандита
- `tests/test_extractor.py` — тесты VLM-экстрактора с моками
- `tests/test_api.py` — интеграционные тесты FastAPI

## Запрещено
- Изменять любые файлы вне `tests/**`
- Делать реальные HTTP-запросы к OpenRouter (только моки)
- Использовать реальный Supabase (использовать тестовую БД или mock)

## Структура тестов

### `tests/conftest.py`
```python
import pytest
from contracts import FEATURE_DIM, N_ARMS
from ml_core.bandits.thompson import ThompsonBandit
import torch

@pytest.fixture
def bandit() -> ThompsonBandit:
    return ThompsonBandit(feature_dim=FEATURE_DIM, n_arms=N_ARMS)

@pytest.fixture
def context() -> torch.Tensor:
    return torch.rand(1, FEATURE_DIM)
```

### `tests/test_thompson.py` — Что тестировать
1. `sample()` возвращает `int` в диапазоне `[0, N_ARMS-1]`
2. `update()` увеличивает матрицу `A` (не равна начальной после update)
3. После многих обновлений с reward=1.0 для arm=0, `sample()` чаще выбирает arm=0
4. Совместимость с GPU если доступен (`bandit.to("cuda")`)

### `tests/test_extractor.py` — Мокирование httpx
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_extract_success():
    mock_response = {
        "choices": [{"message": {"content": '{"is_button_visible": true, "button_color_hex": "#FF0000", "text_sentiment": 0.5, "visual_clutter": 0.3}'}}]
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        # ... вызвать extractor.extract() и проверить UIFeatures
```

### `tests/test_api.py` — FastAPI TestClient
```python
from fastapi.testclient import TestClient
from backend.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
```

## Соглашения
- `pytest-asyncio` для async тестов
- `pytest.mark.asyncio` для корутин
- Покрытие: минимум 80% для `ml_core/`
- Названия тестов: `test_<что_тестируем>_<ожидаемый_результат>`
