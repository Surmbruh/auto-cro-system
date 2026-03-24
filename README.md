# 🤖✨ Auto-CRO System

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Auto-CRO** (Automated Conversion Rate Optimization) — это система для автоматического проведения непрерывных A/B/n тестов и оптимизации UI/UX интерфейсов. 

Вместо классического A/B тестирования система применяет **Contextual Thompson Sampling** (многорукий бандит с контекстом) в связке с **Vision-Language Models (VLM)**. Это позволяет в реальном времени подбирать оптимальный дизайн для каждого клиента, основываясь на "понимании" визуальных характеристик интерфейса.

## 🚀 Ключевые возможности

* 👁️ **Визуальный анализ VLM:** Скриншот страницы отправляется в `Gemini 2.0 Flash` (via OpenRouter), где извлекаются 4 контекстных признака: наличие кнопки CTA, яркость её цвета (HEX), тональность текста и визуальная перегруженность.
* 🎰 **Thompson Bandit (PyTorch):** Обучение байесовских матриц $A$ и $b$ на лету с минимальными затратами CPU-памяти.
* 🕸️ **Легковесный Frontend-SDK:** Vanilla JS скрипт, делающий скриншот DOM-дерева (`html2canvas`) и мгновенно переключающий выигрышный CSS-контекст. Устранена утечка памяти сессий (TTLCache).
* 📊 **Локальный MLflow Tracking:** Готовые дашборды для мониторинга конверсии (`reward`) и снижения неопределённости бандита (`uncertainty`).
* ☁️ **Supabase Persistence:** Веса бандита безопасно дампируются из PyTorch тензоров в Supabase в формате `base64`.
* 🐳 **Docker-Ready:** Оптимизированный CPU-only контейнер для ультра-дешевого продакшен деплоя. 

---

## 🏗️ Архитектура (Data Flow)

1. Пользователь заходит на сайт. Встроенный **SDK** делает скриншот и шлёт в `POST /decide`.
2. **FastAPI** передаёт картинку в **VLM** (OpenRouter `google/gemini-2.0-flash-001`).
3. VLM отвечает строгим **Pydantic** JSON с фичами.
4. Tensor контекста подаётся в PyTorch **Bandit**, который семплирует лучший вариант (`arm_index`) и генерирует уникальный `session_id`.
5. Вариант отображается пользователю. Ему даётся 1 час на нажатие целевой кнопки.
6. В случае клика SDK шлет `POST /feedback` с `session_id`. Матрицы Бандита в **ОЗУ** + **Supabase** синхронизируются, а метрики пишутся в **MLflow**.

---

## 🛠️ Запуск проекта локально

Инструмент сборки: [astral-sh/uv](https://github.com/astral-sh/uv) (в 10-100 раз быстрее pip).

### 1. Переменные окружения
Создайте файл `.env` в корне проекта:
```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ
SUPABASE_URL=https://ваш-проект.supabase.co
SUPABASE_KEY=ваш-секретный-service-key
```

### 2. Установка и запуск (Локально)
```bash
# Установка всех зависимостей проекта (Python 3.13)
uv sync

# Запуск FastAPI бэкенда с горячей перезагрузкой
uv run uvicorn backend.app:app --reload
```
API будет доступен по адресу: `http://127.0.0.1:8000/docs`. <br>
Демо-стенд с внедренным SDK доступен по адресу: `http://127.0.0.1:8000/`.

### 3. Мониторинг метрик (MLflow)
В новой вкладке терминала запустите панель дашбордов:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Панель: `http://127.0.0.1:5000/`.

---

## 🐳 Запуск через Docker Compose

Для деплоя на продакшн сервер (без привязки к локальному `uv` или Python):

```bash
docker compose up --build -d
```
> **Внимание:** В `Dockerfile` специально вшит `torch --extra-index-url https://download.pytorch.org/whl/cpu`, чтобы ужать вес контейнера до разумных 300МБ.

---

## 🧪 Тестирование
```bash
uv run pytest tests/
```
Покрытие: 100% юнит-тестов Thompson-бандита на корректную сходимость (приоритезация выигрышной "ручки") и обработку моковых Async VLM ответов.

---
*Developed with Advanced Agentic Coding Architecture.*
