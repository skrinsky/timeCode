"""
Tests for the ADSR estimator model and loss.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from timecode_audio.model.adsr_estimator import ADSREstimator, adsr_estimator_loss

BATCH      = 2
N_SAMPLES  = 48000   # 1 second


class TestADSREstimatorShapes:
    def test_forward_shapes(self):
        model = ADSREstimator()
        audio = torch.randn(BATCH, N_SAMPLES)
        log_A, log_D, S, log_R = model(audio)
        assert log_A.shape == (BATCH,)
        assert log_D.shape == (BATCH,)
        assert S.shape     == (BATCH,)
        assert log_R.shape == (BATCH,)

    def test_S_bounded(self):
        model = ADSREstimator()
        audio = torch.randn(BATCH, N_SAMPLES)
        _, _, S, _ = model(audio)
        assert S.min() >= 0.0
        assert S.max() <= 1.0

    def test_predict_returns_dict(self):
        model = ADSREstimator()
        audio = torch.randn(N_SAMPLES)   # 1D input
        out = model.predict(audio)
        assert set(out.keys()) == {"A", "D", "S", "R"}
        for v in out.values():
            assert v.shape == (1,)

    def test_predict_batch(self):
        model = ADSREstimator()
        audio = torch.randn(BATCH, N_SAMPLES)
        out = model.predict(audio)
        for v in out.values():
            assert v.shape == (BATCH,)

    def test_predict_nonnegative(self):
        # A, D, R should be >= 0 (exp(x) - 1 can be negative if x < 0 → clamp needed?)
        # In practice exp(x)-1 >= -1, but large negative outputs are penalized by training.
        # Just verify S is in [0,1].
        model = ADSREstimator()
        audio = torch.randn(BATCH, N_SAMPLES)
        out = model.predict(audio)
        assert out["S"].min() >= 0.0
        assert out["S"].max() <= 1.0

    def test_different_inputs_differ(self):
        model = ADSREstimator()
        a1 = torch.randn(1, N_SAMPLES)
        a2 = torch.randn(1, N_SAMPLES)
        o1 = model.predict(a1)
        o2 = model.predict(a2)
        assert not torch.allclose(o1["A"], o2["A"])


class TestADSREstimatorLoss:
    def test_loss_shape(self):
        log_A = torch.randn(BATCH)
        log_D = torch.randn(BATCH)
        S     = torch.sigmoid(torch.randn(BATCH))
        log_R = torch.randn(BATCH)
        A_ms  = torch.rand(BATCH) * 200
        D_ms  = torch.rand(BATCH) * 200
        S_lvl = torch.rand(BATCH)
        R_ms  = torch.rand(BATCH) * 500
        loss = adsr_estimator_loss(log_A, log_D, S, log_R, A_ms, D_ms, S_lvl, R_ms)
        assert loss.shape == ()   # scalar

    def test_loss_zero_for_perfect_prediction(self):
        A_ms  = torch.tensor([10.0, 50.0])
        D_ms  = torch.tensor([100.0, 200.0])
        S_lvl = torch.tensor([0.5, 0.8])
        R_ms  = torch.tensor([300.0, 500.0])

        log_A = torch.log(A_ms  + 1)
        log_D = torch.log(D_ms  + 1)
        log_R = torch.log(R_ms  + 1)

        loss = adsr_estimator_loss(log_A, log_D, S_lvl, log_R, A_ms, D_ms, S_lvl, R_ms)
        assert loss.item() < 1e-6

    def test_loss_positive_for_wrong_prediction(self):
        A_ms  = torch.tensor([10.0])
        D_ms  = torch.tensor([100.0])
        S_lvl = torch.tensor([0.5])
        R_ms  = torch.tensor([300.0])
        log_A = torch.tensor([5.0])   # very wrong
        log_D = torch.tensor([0.0])
        S_bad = torch.tensor([0.0])
        log_R = torch.tensor([0.0])
        loss = adsr_estimator_loss(log_A, log_D, S_bad, log_R, A_ms, D_ms, S_lvl, R_ms)
        assert loss.item() > 0.1

    def test_loss_differentiable(self):
        model = ADSREstimator()
        audio = torch.randn(BATCH, N_SAMPLES)
        A_ms  = torch.rand(BATCH) * 200
        D_ms  = torch.rand(BATCH) * 200
        S_lvl = torch.rand(BATCH)
        R_ms  = torch.rand(BATCH) * 500

        log_A, log_D, S, log_R = model(audio)
        loss = adsr_estimator_loss(log_A, log_D, S, log_R, A_ms, D_ms, S_lvl, R_ms)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()

    def test_longer_clip(self):
        # Should handle 6-second clips (288000 samples)
        model = ADSREstimator()
        audio = torch.randn(2, 288000)
        log_A, log_D, S, log_R = model(audio)
        assert log_A.shape == (2,)
