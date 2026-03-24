---
name: Agent ML
description: Implements ML Core - Thompson Bandit, VLM Feature Extractor, and main pipeline orchestration.
---

# Agent ML — Инструкция

## Твоя зона ответственности
- `ml_core/vlm/extractor.py` — VLM клиент
- `ml_core/bandits/thompson.py` — бандит
- `ml_core/storage.py` — сериализация в Supabase
- `ml_core/mlops.py` — WandB логирование
- `main_pipeline.py` — оркестратор

## Запрещено
- Трогать `backend/**`, `tests/**`
- Изменять `contracts.py` (только читать)
- Использовать `requests` — только `httpx.AsyncClient`
- Использовать `print()` — только `logging`

## Критические контракты (из `contracts.py`)
```python
from contracts import (
    FEATURE_DIM,   # = 4
    N_ARMS,        # = 3
    ContextTensor, # torch.Tensor [1, FEATURE_DIM]
    ArmIndex,      # int
    Reward,        # float
    BanditState,   # TypedDict для Supabase
    StepLog,       # TypedDict для WandB
)
```

## Задачи в порядке приоритета

### 1. FIX Bug — `ml_core/vlm/extractor.py`
`response.json()` вызывается вне `async with` блока — соединение уже закрыто.
**Решение:** перенести строки 74-78 внутрь блока `async with`:
```python
async with httpx.AsyncClient() as client:
    response = await client.post(...)
    response.raise_for_status()
    response_data = response.json()          # ← сюда
    content_str = response_data["choices"][0]["message"]["content"]
    return UIFeatures.model_validate_json(content_str)  # ← сюда
```

### 2. Implement `main_pipeline.py`
Закрыть все 3 TODO:
```python
# 1. Вызвать extractor.extract(image_bytes) → UIFeatures
features: UIFeatures = await extractor.extract(image_bytes)

# 2. Собрать тензор [1, FEATURE_DIM] из UIFeatures
hex_brightness = int(features.button_color_hex.lstrip("#"), 16) / 0xFFFFFF
context = torch.tensor([[
    float(features.is_button_visible),
    hex_brightness,
    features.text_sentiment,
    features.visual_clutter,
]], dtype=torch.float32)

# 3. Передать в бандит
chosen_arm: ArmIndex = bandit.sample(context)
return chosen_arm, context
```

### 3. Implement `ml_core/storage.py`
Смотри скелет файла. Главная логика — `torch.save()` в `io.BytesIO`, encode в base64, сохранить в Supabase таблицу `bandit_states`.

### 4. Implement `ml_core/mlops.py`
Смотри скелет файла. Функция `log_step(step_log: StepLog)` вызывает `wandb.log(...)`.

### 5. Wire up `ThompsonBandit.sync_with_db()`
Вызывает функции из `ml_core/storage.py`.

## Соглашения по тензорам
- Контекст всегда `[1, FEATURE_DIM]`, dtype `torch.float32`
- Операции с GPU: всегда `non_blocking=True`
- `ThompsonBandit` — `nn.Module`, использует `register_buffer`
