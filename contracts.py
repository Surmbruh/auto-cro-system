"""
contracts.py — Единый источник истины для типов на стыке компонентов.

ВАЖНО: Этот файл изменяется ТОЛЬКО координатором (человеком).
Агенты читают его, но не редактируют.

Архитектурный поток данных:
  Frontend (image_bytes)
    → VLMFeatureExtractor → UIFeatures
    → ThompsonBandit.sample(context) → chosen_arm: ArmIndex
    → Frontend показывает вариант UI
    → Frontend присылает FeedbackPayload
    → ThompsonBandit.update(arm_idx, context, reward)
"""

from __future__ import annotations

from typing import TypeAlias
import torch


# ---------------------------------------------------------------------------
# Primitive aliases
# ---------------------------------------------------------------------------

ArmIndex: TypeAlias = int
"""Индекс выбранной «ручки» бандита, от 0 до n_arms-1."""

Reward: TypeAlias = float
"""Скалярная награда: 1.0 — конверсия состоялась, 0.0 — нет."""

ImageBytes: TypeAlias = bytes
"""Сырые байты изображения (JPEG / PNG)."""

ContextTensor: TypeAlias = torch.Tensor
"""Тензор контекста формы [1, feature_dim], dtype=torch.float32."""


# ---------------------------------------------------------------------------
# Feature dimensions contract
# ---------------------------------------------------------------------------

FEATURE_DIM: int = 4
"""
Размерность вектора признаков UIFeatures.
Поля в порядке сборки тензора:
  0: is_button_visible  (bool → float: 0.0 / 1.0)
  1: button_color_hex   (hex → нормализованная яркость float [0..1])
  2: text_sentiment     (float, [-1.0, 1.0])
  3: visual_clutter     (float, [0.0, 1.0])
Изменение FEATURE_DIM требует пересоздания весов ThompsonBandit.
"""

N_ARMS: int = 3
"""
Количество вариантов UI (ручек бандита).
Изменение N_ARMS требует пересоздания весов ThompsonBandit.
"""


# ---------------------------------------------------------------------------
# API boundary types (используются backend ↔ ml_core)
# ---------------------------------------------------------------------------

# TODO (Agent Backend): Заменить TypedDict на Pydantic-схемы из backend/schemas/
# после реализации backend/schemas/requests.py и backend/schemas/responses.py.
# Текущие TypedDict — временный контракт для первоначальной интеграции.

from typing import TypedDict


class DecisionRequest(TypedDict):
    """
    Запрос от Frontend/API → ML Core.
    Передаётся в эндпоинт POST /decide.
    """
    image_bytes: bytes  # Сырой скриншот UI


class DecisionResponse(TypedDict):
    """
    Ответ от ML Core → Frontend/API.
    """
    arm_index: ArmIndex       # Выбранный вариант UI
    confidence: float         # Максимальное сэмплированное вознаграждение (для дебага)


class FeedbackPayload(TypedDict):
    """
    Фидбек от Frontend → API → ML Core.
    Передаётся в эндпоинт POST /feedback.
    """
    arm_index: ArmIndex       # Какой вариант был показан
    reward: Reward             # 1.0 = конверсия, 0.0 = нет
    # context хранится в памяти бандита по arm_index,
    # поэтому не передаётся повторно (оптимизация трафика)


# ---------------------------------------------------------------------------
# Storage contract (ml_core ↔ Supabase / MLOps)
# ---------------------------------------------------------------------------

class BanditState(TypedDict):
    """
    Сериализованное состояние бандита для хранения в Supabase.
    Таблица: bandit_states. Используется Agent MLOps.
    """
    arm_index: ArmIndex        # Для какой ручки сохраняем матрицы
    a_bytes: str               # base64-encoded bytes от torch.save(A[i])
    b_bytes: str               # base64-encoded bytes от torch.save(b[i])
    updated_at: str            # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# Logging contract (ml_core ↔ WandB)
# ---------------------------------------------------------------------------

class StepLog(TypedDict):
    """
    Запись одного шага для логирования в WandB.
    Используется Agent MLOps в ml_core/mlops.py.
    """
    step: int
    chosen_arm: ArmIndex
    reward: Reward
    uncertainty: float         # sqrt(ctx @ A_inv @ ctx.T) — мера неопределённости
    model_name: str            # Имя VLM-модели, использованной для извлечения фичей
