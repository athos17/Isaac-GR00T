# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re

from gr00t.eval.run_gr00t_server import GetActionInputVisualizationPolicy
import numpy as np
from PIL import Image


class DummyPolicy:
    def __init__(self):
        self.get_action_calls = []
        self.reset_calls = []

    def get_action(self, observation, options=None):
        self.get_action_calls.append((observation, options))
        return {"joint_pos": np.ones(2, dtype=np.float32)}, {"source": "dummy"}

    def reset(self, options=None):
        self.reset_calls.append(options)
        return {"reset": True}

    def get_modality_config(self):
        return {"state": "config"}

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass


def test_get_action_input_visualization_policy_saves_video_pngs_and_summary(tmp_path: Path):
    policy = DummyPolicy()
    wrapped = GetActionInputVisualizationPolicy(policy, tmp_path)

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[..., 0] = 255
    observation = {
        "video": {"front": frame[None, None]},
        "state": {"joint_pos": np.array([[[0.1, 0.2]]], dtype=np.float32)},
        "language": {"task": [["pick up the cube"]]},
    }
    options = {"temperature": 0.0}
    action, info = wrapped.get_action(observation, options)

    assert policy.get_action_calls == [(observation, options)]
    np.testing.assert_array_equal(action["joint_pos"], np.ones(2, dtype=np.float32))
    assert info == {"source": "dummy"}

    request_dirs = list(tmp_path.iterdir())
    assert len(request_dirs) == 1
    request_dir = request_dirs[0]
    assert re.fullmatch(r"get_action_000001_\d{8}_\d{6}_\d{6}", request_dir.name)
    assert request_dir.is_dir()
    image_path = request_dir / "video_front_b000_t000.png"
    assert image_path.is_file()
    image = Image.open(image_path)
    assert image.size == (5, 4)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert (request_dir / "state_joint_pos.png").is_file()
    assert (request_dir / "summary.json").is_file()


def test_get_action_input_visualization_policy_does_not_save_reset_inputs(tmp_path: Path):
    policy = DummyPolicy()
    wrapped = GetActionInputVisualizationPolicy(policy, tmp_path)

    assert wrapped.reset({"episode": 1}) == {"reset": True}

    assert policy.reset_calls == [{"episode": 1}]
    assert list(tmp_path.iterdir()) == []
