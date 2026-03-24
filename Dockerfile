FROM python:3.13-slim

# Настройка окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Копируем быстрый менеджер пакетов UV из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Сначала копируем исходники и конфиги
COPY . .

# Инициализируем виртуальное окружение
RUN uv venv

# ВАЖНО: Устанавливаем torch явно для CPU-only. 
# Бинарники PyTorch с CUDA могут весить до 2 ГБ, а для продакшена (так как модель ML легковесна)
# нам нужен только CPU-вариант (он весит ~200 МБ).
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Читаем зависимости напрямую из pyproject.toml 
# (точка означает установку текущего проекта и его зависимостей)
RUN uv pip install .

# Экспонируем порт
EXPOSE 8000

# Запускаем Uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
