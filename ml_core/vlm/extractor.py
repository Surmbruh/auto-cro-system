import httpx
import base64
from pydantic import BaseModel, Field

class UIFeatures(BaseModel):
    """Схема фичей, которые VLM должна извлечь из скриншота."""
    is_button_visible: bool = Field(description="Видна ли главная кнопка CTA (Call to Action)")
    button_color_hex: str = Field(description="HEX цвет кнопки, например #FF5733")
    text_sentiment: float = Field(description="Тональность текста от -1.0 (негатив) до 1.0 (позитив)")
    visual_clutter: float = Field(description="Степень визуальной перегруженности от 0.0 до 1.0")

class VLMFeatureExtractor:
    """
    Асинхронный клиент для извлечения признаков из UI через OpenRouter API.
    """
    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-001"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def extract(self, image_bytes: bytes) -> UIFeatures:
        """
        Отправляет скриншот в VLM и возвращает структурированные признаки.
        
        Args:
            image_bytes: Сырые байты изображения (JPEG/PNG)
            
        Returns:
            UIFeatures: Валидированный Pydantic-объект
        """
        # Закодировать image_bytes в base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{base64_image}"
        
        # Динамически получаем JSON схему Pydantic модели
        schema_json = UIFeatures.model_json_schema()
        
        # Системный промпт, требующий строгий JSON
        prompt = (
            "Проанализируй скриншот UI. Строго верни JSON, который соответствует "
            f"следующей Pydantic схеме (JSON Schema):\n{schema_json}\n"
            "Верни только сам JSON-объект (начиная с фигурной скобки), "
            "не добавляй markdown-разметки или иной текст."
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Выполняем POST запрос к OpenRouter
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url, headers=headers, json=payload, timeout=30.0
                )
                response.raise_for_status()
                
                response_data = response.json()
                content_str = response_data["choices"][0]["message"]["content"]
                
                # Очистка разметки Markdown, так как VLM любят присылать ответ в ```json ... ```
                content_str = content_str.strip()
                if content_str.startswith("```json"):
                    content_str = content_str[7:]
                elif content_str.startswith("```"):
                    content_str = content_str[3:]
                
                if content_str.endswith("```"):
                    content_str = content_str[:-3]
                    
                content_str = content_str.strip()
                
                # Парсинг ответа через строгую валидацию Pydantic
                return UIFeatures.model_validate_json(content_str)
            
        except httpx.HTTPStatusError as e:
            error_details = e.response.text if hasattr(e, 'response') else str(e)
            raise RuntimeError(f"Ошибка вызова OpenRouter API (HTTP {e.response.status_code}): {error_details}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Сетевая ошибка вызова OpenRouter API: {e}")
        except Exception as e:
            raise ValueError(f"Ошибка парсинга или валидации ответа от VLM: {e}")