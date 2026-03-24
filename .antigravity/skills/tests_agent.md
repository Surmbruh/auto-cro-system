---
name: Agent MLOps
description: Implements WandB logging, Supabase state serialization, and MCP server tooling.
---

# Agent MLOps — Инструкция

## Твоя зона ответственности
- `ml_core/mlops.py` — WandB логирование
- `ml_core/storage.py` — сериализация/десериализация состояния бандита
- `mcp_supabase_server.py` — MCP JSON-RPC сервер

## Запрещено
- Трогать `ml_core/bandits/**`, `ml_core/vlm/**`, `backend/**`, `tests/**`
- Изменять `contracts.py`

## Критические контракты (из `contracts.py`)
```python
from contracts import BanditState, StepLog, ArmIndex
```

## Задачи в порядке приоритета

### 1. `ml_core/mlops.py` — WandB логирование
```python
import wandb
from contracts import StepLog

def init_wandb(project: str = "auto-cro", config: dict | None = None) -> None:
    wandb.init(project=project, config=config or {})

def log_step(step_log: StepLog) -> None:
    wandb.log({
        "chosen_arm": step_log["chosen_arm"],
        "reward": step_log["reward"],
        "uncertainty": step_log["uncertainty"],
        "model_name": step_log["model_name"],
    }, step=step_log["step"])

def finish() -> None:
    wandb.finish()
```

### 2. `ml_core/storage.py` — Supabase сериализация
- Тензоры сохранять через `torch.save()` → `io.BytesIO` → `base64.b64encode`
- Таблица Supabase: `bandit_states` с колонками: `arm_index`, `A_bytes`, `b_bytes`, `updated_at`
- `save_state(client, bandit, arm_idx)` — upsert по `arm_index`
- `load_state(client, arm_idx)` → `BanditState` — select по `arm_index`

### 3. `mcp_supabase_server.py` — Инструмент `check_supabase`
Реализовать реальный запрос к Supabase при вызове метода `tools/call` с именем `check_supabase`:
- Подключиться к Supabase через env переменные `SUPABASE_URL`, `SUPABASE_KEY`
- Выполнить `SELECT COUNT(*) FROM bandit_states`
- Вернуть результат в JSON-RPC ответе

## Соглашения
- Supabase credentials только из `os.environ`, никогда хардкод
- Логировать ошибки через `logging`, не `print()`
- При недоступности WandB — graceful degradation, не падать
