import torch
import torch.nn as nn
import typing

class ThompsonBandit(nn.Module):
    """
    Контекстный многорукий бандит, использующий Linear Thompson Sampling.
    Поддерживает батчинг и вычисления на GPU (через PyTorch).
    """
    def __init__(self, feature_dim: int, n_arms: int, lambda_: float = 1.0):
        super().__init__()
        self.d = feature_dim
        self.k = n_arms
        
        # Инициализация параметров: A (матрица точности) и b (вектор наград)
        self.register_buffer("A", torch.eye(self.d).repeat(self.k, 1, 1) * lambda_)
        self.register_buffer("b", torch.zeros(self.k, self.d, 1))
        
    def sample(self, context: torch.Tensor) -> int:
        """
        Выбирает оптимальную ручку на основе контекста.
        
        Args:
            context: Тензор признаков размерности [1, d]
            
        Returns:
            int: Индекс выбранной ручки (от 0 до n_arms - 1)
        """
        A = typing.cast(torch.Tensor, self.A)
        b = typing.cast(torch.Tensor, self.b)
        ctx = context.to(device=A.device, non_blocking=True)
        
        # Инвертируем батчи матриц A: [k, d, d]
        A_inv = torch.linalg.inv(A)
        
        # Вычисляем среднее theta = A^-1 b: [k, d, d] @ [k, d, 1] -> [k, d, 1] -> [k, d]
        theta_mean = torch.matmul(A_inv, b).squeeze(-1)
        
        # Сэмплируем theta для каждой ручки
        dist = torch.distributions.MultivariateNormal(
            loc=theta_mean,
            covariance_matrix=A_inv
        )
        theta_sample = dist.sample() # [k, d]
        
        # Ожидаемая награда для каждой ручки: context @ theta_sample^T
        # ctx: [1, d], theta_sample.T: [d, k] => [1, k] => [k]
        expected_rewards = torch.matmul(ctx, theta_sample.T).squeeze(0)
        
        return int(torch.argmax(expected_rewards).item())

    def update(self, arm_idx: int, context: torch.Tensor, reward: float) -> None:
        """
        Обновляет апостериорное распределение для выбранной ручки.
        
        Args:
            arm_idx: Индекс ручки, которую потянули
            context: Тензор признаков размерности [1, d]
            reward: Полученная награда (например, 1.0 за клик, 0.0 за игнор)
        """
        A = typing.cast(torch.Tensor, self.A)
        b = typing.cast(torch.Tensor, self.b)
        # Переносим контекст на устройство (напр., GPU)
        x = context.view(self.d, 1).to(device=A.device, non_blocking=True)
        r = torch.tensor(reward, dtype=b.dtype, device=A.device)
        
        # Обновление матрицы точности: A = A + x*x^T
        A[arm_idx] += torch.matmul(x, x.T)
        
        # Обновление вектора наград: b = b + r*x
        b[arm_idx] += r * x

    def sync_with_db(self, mcp_client: typing.Any) -> None:
        """
        Синхронизация весов с Supabase через MCP.
        """
        from ml_core.storage import save_state
        A = typing.cast(torch.Tensor, self.A)
        b = typing.cast(torch.Tensor, self.b)
        for i in range(self.k):
            save_state(mcp_client, A[i], b[i], arm_idx=i)