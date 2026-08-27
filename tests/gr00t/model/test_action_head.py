# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.embodiment_conditioned_mlp import MultiEmbodimentActionEncoder
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])

    def test_training_rtc_conditioning_prefix_and_shared_tau(self):
        config = _small_config(
            training_rtc_enabled=True,
            training_rtc_max_delay=2,
            training_rtc_delay_pmf={0: 0.0, 1: 1.0, 2: 0.0},
        )
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        action_mask = torch.ones_like(actions)
        noisy, noise, token_t, loss_mask, delays = head._build_training_rtc_conditioning(
            actions, action_mask
        )
        assert torch.equal(delays, torch.ones_like(delays))
        assert torch.equal(noisy[:, 0], actions[:, 0])
        assert torch.equal(loss_mask[:, 0], torch.zeros_like(loss_mask[:, 0]))
        assert torch.all(token_t[:, 0] == 1.0)
        assert torch.allclose(token_t[:, 1:], token_t[:, 1:2])
        assert torch.all(loss_mask[:, 1:] == 1.0)

    def test_training_rtc_supports_delay_eleven(self):
        config = _small_config(
            training_rtc_enabled=True,
            training_rtc_max_delay=11,
            training_rtc_delay_pmf={11: 1.0},
            action_horizon=32,
        )
        head = Gr00tN1d7ActionHead(config)
        delays = head._sample_training_rtc_delay(batch_size=16, device=torch.device("cpu"))
        assert torch.equal(delays, torch.full((16,), 11, dtype=torch.long))

    def test_training_rtc_d0_has_no_prefix_mask(self):
        config = _small_config(
            training_rtc_enabled=True,
            training_rtc_max_delay=0,
            training_rtc_delay_pmf={0: 1.0},
        )
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        action_mask = torch.ones_like(actions)
        noisy, _, token_t, loss_mask, delays = head._build_training_rtc_conditioning(
            actions, action_mask
        )
        assert torch.equal(delays, torch.zeros_like(delays))
        assert torch.all(loss_mask == action_mask)
        assert torch.all(token_t == token_t[:, :1])
        assert not torch.equal(noisy, actions)

    def test_training_rtc_d0_matches_legacy_noise_and_tau_under_fixed_rng(self):
        config = _small_config(
            training_rtc_enabled=True,
            training_rtc_max_delay=0,
            training_rtc_delay_pmf={0: 1.0},
        )
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        mask = torch.ones_like(actions)
        torch.manual_seed(1234)
        rtc_noisy, rtc_noise, rtc_t, _, _ = head._build_training_rtc_conditioning(actions, mask)
        torch.manual_seed(1234)
        legacy_noise = torch.randn_like(actions)
        legacy_tau = head.sample_time(2, actions.device, actions.dtype)
        legacy_noisy = legacy_tau[:, None, None] * actions + (1 - legacy_tau[:, None, None]) * legacy_noise
        torch.testing.assert_close(rtc_noise, legacy_noise)
        torch.testing.assert_close(rtc_t[:, 0], legacy_tau)
        torch.testing.assert_close(rtc_noisy, legacy_noisy)


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1

    def test_training_rtc_sampler_hard_overwrites_prefix(self):
        config = _small_config(
            training_rtc_enabled=True,
            training_rtc_max_delay=2,
            num_inference_timesteps=2,
        )
        head = Gr00tN1d7ActionHead(config).eval()
        action_input = _make_action_input(config, batch_size=1)
        committed = action_input["action"].clone()
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
            options={"rtc_mode": "training", "d_cond": 2},
        )
        assert torch.equal(out["action_pred"][:, :2], committed[:, :2])

    def test_training_rtc_sampler_rejects_out_of_distribution_delay(self):
        config = _small_config(training_rtc_enabled=True, training_rtc_max_delay=1)
        head = Gr00tN1d7ActionHead(config).eval()
        action_input = _make_action_input(config, batch_size=1)
        with pytest.raises(RuntimeError, match="RTC_DELAY_OOD"):
            head.get_action(
                _make_backbone_output(config, batch_size=1),
                action_input,
                options={"rtc_mode": "training", "d_cond": 2},
            )


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad


class TestTrainingRTCActionTimestep:
    def test_action_encoder_accepts_tokenwise_timesteps(self):
        encoder = MultiEmbodimentActionEncoder(action_dim=5, hidden_size=8, num_embodiments=2)
        actions = torch.randn(2, 4, 5)
        cat_ids = torch.zeros(2, dtype=torch.long)
        scalar = torch.tensor([10, 20], dtype=torch.long)
        tokenwise = torch.tensor([[10, 10, 20, 20], [20, 20, 10, 10]], dtype=torch.long)

        scalar_out = encoder(actions, scalar, cat_ids)
        tokenwise_out = encoder(actions, tokenwise, cat_ids)

        assert scalar_out.shape == tokenwise_out.shape == (2, 4, 8)
        assert not torch.allclose(tokenwise_out[:, 0], tokenwise_out[:, 2])

    def test_action_encoder_rejects_wrong_timestep_shape(self):
        encoder = MultiEmbodimentActionEncoder(action_dim=5, hidden_size=8, num_embodiments=2)
        with pytest.raises(ValueError, match="Expected timesteps shape"):
            encoder(
                torch.randn(2, 4, 5),
                torch.zeros(2, 3),
                torch.zeros(2, dtype=torch.long),
            )
