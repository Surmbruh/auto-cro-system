import torch
import asyncio
from ml_core.vlm.extractor import VLMFeatureExtractor, UIFeatures
from ml_core.bandits.thompson import ThompsonBandit

async def run_optimization_step(api_key: str, image_bytes: bytes) -> tuple[int, torch.Tensor]:
    """
    Полный цикл: Парсинг картинки -> Извлечение фичей -> Выбор дизайна бандитом.
    """
    # 1. Инициализация компонентов
    extractor = VLMFeatureExtractor(api_key=api_key)
    # 4 фичи из UIFeatures, 3 варианта дизайна (arms)
    bandit = ThompsonBandit(feature_dim=4, n_arms=3) 

    # 2. Извлечение признаков (Vision)
    features: UIFeatures = await extractor.extract(image_bytes)
    
    # 3. Преобразование Pydantic-модели в PyTorch-тензор
    hex_brightness = int(features.button_color_hex.lstrip("#"), 16) / 0xFFFFFF
    context_tensor = torch.tensor([[
        float(features.is_button_visible),
        hex_brightness,
        features.text_sentiment,
        features.visual_clutter,
    ]], dtype=torch.float32)
    
    # 4. Принятие решения (Decision)
    chosen_arm = bandit.sample(context_tensor)
    
    # 5. Возврат результата
    return chosen_arm, context_tensor

if __name__ == "__main__":
    import os
    
    async def main() -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "dummy_key")
        # Dummy test data instead of a real image to prevent test hanging if no disk image
        dummy_image = b"dummy_image_bytes_for_testing"
        print(f"Starting test run_optimization_step with key={api_key[:4]}...")
        try:
            arm, ctx = await run_optimization_step(api_key, dummy_image)
            print(f"Success! Chosen arm: {arm}, Context: {ctx}")
        except Exception as e:
            print(f"Exception during test run: {e}")
            
    asyncio.run(main())