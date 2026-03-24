Приветствую. Как Senior AI Architect, я подготовил для вас структурный документ по проектированию системы Auto-CRO на базе алгоритма Contextual Thompson Sampling, опираясь на архитектурные паттерны, эффективную работу с GPU и интеграцию с указанным стеком.

### 1. Математическая модель: Обновление весов (Contextual Thompson Sampling)

В нашей задаче каждый вариант веб-интерфейса (UI) представляется в виде $d$-мерного вектора фичей (контекста) $b_i(t) \in \mathbb{R}^d$, где $i$ — номер варианта, а $t$ — временной шаг взаимодействия. Предполагается, что существует неизвестный вектор параметров $\mu \in \mathbb{R}^d$, определяющий ожидаемую награду (например, конверсию) как $b_i(t)^T \mu$.

Мы используем нормальное (гауссовское) распределение для правдоподобия и априорных вероятностей. 

**Формулы байесовского обновления (Posterior Update)**:
На шаге $t$ мы храним матрицу ковариации $B(t)$ и вектор $f(t)$ для вычисления оценки параметров $\hat{\mu}(t)$. 

1. **Априорное распределение** (Prior) параметров $\mu$ на шаге $t$ моделируется как:
   $\mathcal{N}(\hat{\mu}(t), v^2 B(t)^{-1})$
   где $v$ — гиперпараметр (масштаб дисперсии).

2. **Оценка параметров**:
   $B(t) = I_d + \sum_{\tau=1}^{t-1} b_{a(\tau)}(\tau) b_{a(\tau)}(\tau)^T$
   $\hat{\mu}(t) = B(t)^{-1} \left( \sum_{\tau=1}^{t-1} b_{a(\tau)}(\tau) r_{a(\tau)}(\tau) \right)$

3. **Обновление после наблюдения (Posterior)**:
   При выборе варианта интерфейса $a(t)$ и получении награды $r_t$ (например, клика), параметры на шаге $t+1$ обновляются рекурсивно:
   $B = B + b_{a(t)}(t) b_{a(t)}(t)^T$
   $f = f + b_{a(t)}(t) r_t$
   $\hat{\mu} = B^{-1} f$

Апостериорное распределение на шаге $t+1$ примет вид $\mathcal{N}(\hat{\mu}(t+1), v^2 B(t+1)^{-1})$. На каждом шаге мы сэмплируем вектор $\tilde{\mu}(t)$ из этого распределения и выбираем вариант UI, максимизирующий $b_i(t)^T \tilde{\mu}(t)$.

---

### 2. Проектирование API-клиента: Извлечение фичей через OpenRouter API

Для извлечения эмбеддингов мы отправляем скриншот (в формате base64) в мультимодальную LLM через OpenRouter API. Для строгой валидации структуры JSON-ответа мы используем Pydantic модели, применяя метод `model_validate_json`. OpenRouter поддерживает параметр `response_format: { type: 'json_schema' }` для гарантии структурированного ответа.

```python
import base64
from typing import List
from pydantic import BaseModel, Field
import requests

# Pydantic модель для строгой валидации ответа LLM
class UIEmbeddings(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Вектор эмбеддингов UI фичей, извлеченных из скриншота и DOM-дерева"
    )

class VLMFeatureExtractor:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _encode_image(self, image_path: str) -> str:
        # Кодируем изображение в base64
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

    def extract_features(self, image_path: str, dom_tree: str) -> UIEmbeddings:
        base64_image = self._encode_image(image_path)
        
        payload = {
            "model": self.model, #
            "response_format": {"type": "json_object"}, # Включаем JSON-режим
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Проанализируй DOM-дерево и UI скриншот. Верни JSON с ключом 'features', содержащий массив чисел с плавающей точкой.\nDOM: {dom_tree}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image} # Передача base64 изображения
                        }
                    ]
                }
            ],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_json_str = response.json()["choices"]["message"]["content"] #
        
        # Строгая валидация Pydantic
        validated_data = UIEmbeddings.model_validate_json(response_json_str)
        return validated_data
```

---

### 3. Архитектура ML-ядра: ThompsonBandit на PyTorch

Поскольку операции на GPU по умолчанию асинхронны, мы должны использовать флаг `non_blocking=True` при перемещении тензоров между CPU и GPU, чтобы избежать блокировок (CUDA sync), которые могут снизить пропускную способность нашего FastAPI-приложения.

```python
import torch

class ThompsonBandit:
    def __init__(self, d: int, v: float, device: str = "cuda"):
        self.d = d
        self.v = v
        self.device = torch.device(device)
        
        # Инициализация параметров по алгоритму TS: B = Id, f = 0d, mu_hat = 0d
        self.B = torch.eye(d, device=self.device)
        self.f = torch.zeros((d, 1), device=self.device)
        self.mu_hat = torch.zeros((d, 1), device=self.device)

    def sample_and_select(self, contexts: torch.Tensor) -> int:
        """
        contexts: Тензор формы [N, d], где N - количество UI-вариантов
        Возвращает: Индекс выбранного варианта
        """
        # Переносим контекст на GPU без блокировки основного потока FastAPI
        contexts = contexts.to(self.device, non_blocking=True)
        
        # Обращение ковариационной матрицы
        B_inv = torch.linalg.inv(self.B)
        covariance_matrix = (self.v ** 2) * B_inv
        
        # Сэмплируем mu_tilde из распределения N(mu_hat, v^2 * B^-1)
        dist = torch.distributions.MultivariateNormal(
            loc=self.mu_hat.squeeze(), 
            covariance_matrix=covariance_matrix
        )
        mu_tilde = dist.sample().unsqueeze(1)
        
        # Оценка: a(t) = argmax_i b_i(t)^T mu_tilde(t)
        rewards_estimation = torch.matmul(contexts, mu_tilde)
        best_action = torch.argmax(rewards_estimation).item()
        
        return best_action

    def update(self, chosen_context: torch.Tensor, reward: float):
        """
        Обновление параметров после получения фидбека (награды)
        chosen_context: Тензор формы [d, 1]
        """
        # Используем non_blocking=True для предотвращения CUDA-синхронизации, 
        # скрывая overhead CPU
        b = chosen_context.to(self.device, non_blocking=True)
        r = torch.tensor([[reward]], device=self.device)

        # Обновление B = B + b * b^T
        self.B.add_(torch.matmul(b, b.T))
        
        # Обновление f = f + b * r
        self.f.add_(b * r)
        
        # Пересчет mu_hat = B^-1 * f
        B_inv = torch.linalg.inv(self.B)
        self.mu_hat = torch.matmul(B_inv, self.f)
```

---

### 4. MLOps стратегия: Логирование в Weights & Biases (WandB)

В дополнение к базовым метрикам (CTR, Regret), для глубокого анализа контекстного алгоритма через WandB нам необходимо логировать следующую структуру:

**1. Системные и конфигурационные параметры:**
* **System metrics**: Загрузка CPU/GPU и сети. Это критично для FastAPI-сервера, так как блокирующие CUDA-вызовы могут оставлять GPU в простое.
* **Hyperparameters**: Конфигурация модели (параметр $v$, размерность $d$, параметры API) должна сохраняться через `wandb.init(config=...)`.

**2. Метрики исследования среды (Exploration metrics):**
* **Standard Deviation (Неопределенность)**: Стандартное отклонение оценок (математически $s_{t,i} = \sqrt{b_i(t)^T B(t)^{-1} b_i(t)}$). Эта метрика показывает, насколько алгоритм "не уверен" в конкретном варианте интерфейса. Разделение UI-вариантов на "насыщенные" и "ненасыщенные" (saturated/unsaturated) поможет понять, достаточно ли алгоритм исследует среду.

**3. Интерактивные данные (Media):**
* **Скриншоты интерфейсов (Datasets/Media)**: Необходимо логировать изображения UI-компонентов, которые побеждают (лучшие по метрикам) и проигрывают на каждом шаге, используя инструменты логирования медиа (например, `wandb.Image()`).

**4. Сводные метрики (Summary metrics):**
* Трекинг лучших достигнутых значений за сессию (настраивается через `wandb.run.summary["best_..."]` или `define_metric` с аргументами `"max"`, `"min"`, `"best"`). Это позволит в дашборде WandB быстро сравнивать разные эксперименты (run'ы) по максимальной конверсии, а не только по финальной.

---

### 5. MLOps Architecture

**1. Схема метрик WandB (Experiment Tracking)**
Каждый шаг взаимодействия пользователя с интерфейсом (выбор ручки и получение награды) логируется как отдельный `step` в WandB:
* `chosen_arm`: Индекс выбранного варианта интерфейса (int)
* `reward`: Сигнал награды за шаг (float, например, `0.0` или `1.0`)
* `uncertainty`: Количественная оценка "неуверенности" алгоритма для выбранного варианта (float, вычисляемая на базе аппроксимации).
* `wandb.run.summary["best_reward"]`: Динамически обновляемая метрика максимальной награды, зафиксированной в рамках одной сессии запуска, для быстрой оценки эффективности выбранной стратегии и конфигурации.

**2. Схема персистентности Supabase (`bandit_states`)**
Для хранения состояния бандита между вызовами сервера используется реляционная таблица в PostgreSQL:
* `arm_index` (int, `PRIMARY KEY`): Уникальный индекс ручки
* `A_bytes` (text): Сериализованная матрица точности $A$ (base64 кодировка байтов PyTorch-тензора)
* `b_bytes` (text): Сериализованный вектор наград $b$ (base64 кодировка)
* `updated_at` (timestamptz): Время последнего обновления
* **Стратегия: `upsert`** на основе уникального индекса `arm_index`. При загрузке состояния (load) тензоры десериализуются из `base64` в нативные объекты `torch.Tensor`.

**3. Graceful Degradation (Отказоустойчивость)**
При недоступности или ошибках инициализации и логирования во внешних сервисах (например, WandB не установлен, отсутствует соединение с Supabase), наша архитектура избегает краша системы (Crash):
* Выбрасываемые исключения перехватываются внутри `try/except` блоков.
* Информация об ошибке фиксируется через модуль `logging` (`logger.warning`).
* Функциональность MLOps (логирование/сохранение состояния) отключается для текущего вызова, но основной поток ответа (сэмплирование варианта для фронтенда и прием фидбека) продолжает работать без прерываний.
