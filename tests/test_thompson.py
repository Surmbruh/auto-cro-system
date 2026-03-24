"""
tests/test_thompson.py — Unit tests for ThompsonBandit.

Tests: sample(), update(), posterior convergence, GPU compatibility.
Agent Tests owns this file.
"""
from __future__ import annotations

import torch
import pytest
from collections import Counter

from contracts import FEATURE_DIM, N_ARMS
from ml_core.bandits.thompson import ThompsonBandit


class TestBanditSample:
    """Tests for ThompsonBandit.sample()."""

    @pytest.mark.parametrize("dim, arms", [(2, 2), (4, 3), (8, 5)])
    def test_sample_returns_valid_arm_index_parametrized(self, dim: int, arms: int) -> None:
        """sample() should return an int in [0, arms-1] for various dimensions and arm counts."""
        bandit = ThompsonBandit(feature_dim=dim, n_arms=arms)
        context = torch.rand(1, dim, dtype=torch.float32)
        arm = bandit.sample(context)
        assert isinstance(arm, int)
        assert 0 <= arm < arms

    def test_sample_returns_valid_arm_index(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """sample() should return an int in [0, N_ARMS-1]."""
        arm = bandit.sample(context)
        assert isinstance(arm, int)
        assert 0 <= arm < N_ARMS

    def test_sample_is_deterministic_given_seed(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """With same seed, sample() should return same arm."""
        torch.manual_seed(42)
        arm1 = bandit.sample(context)
        torch.manual_seed(42)
        arm2 = bandit.sample(context)
        assert arm1 == arm2

    def test_sample_explores_all_arms(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """With no updates, bandit should explore all arms over many samples."""
        counts = Counter(bandit.sample(context) for _ in range(300))
        # All arms should be sampled at least once with high probability
        assert len(counts) == N_ARMS, f"Only sampled arms: {dict(counts)}"


class TestBanditUpdate:
    """Tests for ThompsonBandit.update()."""

    def test_update_changes_A_matrix(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """update() should modify A[arm_idx] away from identity."""
        A_before = bandit.A[0].clone()
        bandit.update(arm_idx=0, context=context, reward=1.0)
        assert not torch.allclose(bandit.A[0], A_before), "A matrix should change after update"

    def test_update_changes_b_vector(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """update() with reward=1.0 should make b[arm_idx] non-zero."""
        bandit.update(arm_idx=0, context=context, reward=1.0)
        assert not torch.allclose(bandit.b[0], torch.zeros_like(bandit.b[0]))

    def test_update_zero_reward_leaves_b_unchanged(
        self, bandit: ThompsonBandit, context: torch.Tensor
    ) -> None:
        """update() with reward=0.0 should not change b (b += 0 * x = 0)."""
        b_before = bandit.b[0].clone()
        bandit.update(arm_idx=0, context=context, reward=0.0)
        assert torch.allclose(bandit.b[0], b_before)

    def test_update_only_affects_chosen_arm(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """update() on arm 0 should NOT change A[1] or A[2]."""
        A1_before = bandit.A[1].clone()
        A2_before = bandit.A[2].clone()
        bandit.update(arm_idx=0, context=context, reward=1.0)
        assert torch.allclose(bandit.A[1], A1_before)
        assert torch.allclose(bandit.A[2], A2_before)

    def test_update_increases_A_more_than_unupdated_arm(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """Updating arm 0 N times should make its A matrix 'larger' (e.g., in trace) than arm 1."""
        for _ in range(5):
            bandit.update(arm_idx=0, context=context, reward=1.0)
        
        trace_0 = torch.trace(bandit.A[0])
        trace_1 = torch.trace(bandit.A[1])
        assert trace_0 > trace_1


class TestBanditConvergence:
    """Tests for posterior convergence behaviour."""

    def test_bandit_prefers_rewarded_arm(self, bandit: ThompsonBandit, zero_context: torch.Tensor) -> None:
        """After many rewarded updates on arm 0, bandit should prefer arm 0."""
        # Update arm 0 heavily
        for _ in range(100):
            bandit.update(arm_idx=0, context=zero_context, reward=1.0)
        
        # Sample 200 times
        samples = [bandit.sample(zero_context) for _ in range(200)]
        counts = Counter(samples)
        
        # Arm 0 should be chosen >70% of the time (>140 times)
        assert counts[0] > 140

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bandit_moves_to_gpu(self, bandit: ThompsonBandit, context: torch.Tensor) -> None:
        """Bandit should work correctly on GPU."""
        bandit = bandit.to("cuda")
        arm = bandit.sample(context)
        assert 0 <= arm < N_ARMS
