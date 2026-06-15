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

from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "unnamed"


def _json_summary(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            if finite.size > 0:
                summary["min"] = float(finite.min())
                summary["max"] = float(finite.max())
                summary["mean"] = float(finite.mean())
        if value.size <= 32:
            summary["values"] = value.tolist()
        return summary
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_summary(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _as_uint8_image(value: Any) -> Any | None:
    import numpy as np

    image = np.asarray(value)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.ndim != 2:
        return None

    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    if image.size == 0:
        return None
    image = np.nan_to_num(image)
    if image.min() >= 0.0 and image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _iter_video_frames(value: Any):
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 3:
        yield "", array
        return
    if array.ndim == 4 and array.shape[-1] in (1, 3, 4):
        for frame_idx in range(array.shape[0]):
            yield f"t{frame_idx:03d}", array[frame_idx]
        return
    if array.ndim >= 5 and array.shape[-1] in (1, 3, 4):
        leading_shape = array.shape[:-3]
        for idx in np.ndindex(leading_shape):
            if len(idx) >= 2:
                suffix = f"b{idx[0]:03d}_t{idx[1]:03d}"
                if len(idx) > 2:
                    suffix += "_" + "_".join(
                        f"i{axis}{value:03d}" for axis, value in enumerate(idx[2:], 2)
                    )
            else:
                suffix = "_".join(f"i{axis}{value:03d}" for axis, value in enumerate(idx))
            yield suffix, array[idx]


def _save_image(path: Path, image: Any) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def _save_state_plot(path: Path, value: Any) -> bool:
    import numpy as np

    array = np.asarray(value)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return False
    y = array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(-1, 1)

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if y.shape[0] == 1:
        ax.bar(np.arange(y.shape[1]), y[0])
        ax.set_xlabel("dimension")
    else:
        ax.plot(y)
        ax.set_xlabel("flattened time index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


class GetActionInputVisualizationPolicy:
    """Policy wrapper that saves visual artifacts for each get_action input."""

    def __init__(self, policy: Any, output_dir: str | Path):
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strict = getattr(policy, "strict", False)
        self._request_index = 0

    def _next_request_dir(self) -> Path:
        while True:
            self._request_index += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            request_dir = self.output_dir / f"get_action_{self._request_index:06d}_{timestamp}"
            if not request_dir.exists():
                request_dir.mkdir(parents=True)
                return request_dir

    def _save_get_action_input(
        self, observation: dict[str, Any], options: dict[str, Any] | None
    ) -> None:
        request_dir = self._next_request_dir()
        summary: dict[str, Any] = {
            "request_index": self._request_index,
            "video": {},
            "state": _json_summary(observation.get("state", {})),
            "language": _json_summary(observation.get("language", {})),
            "options": _json_summary(options),
            "files": [],
        }

        for camera_name, frames in observation.get("video", {}).items():
            safe_camera_name = _safe_filename(str(camera_name))
            summary["video"][str(camera_name)] = _json_summary(frames)
            for suffix, frame in _iter_video_frames(frames):
                image = _as_uint8_image(frame)
                if image is None:
                    continue
                suffix_part = f"_{suffix}" if suffix else ""
                filename = f"video_{safe_camera_name}{suffix_part}.png"
                _save_image(request_dir / filename, image)
                summary["files"].append(filename)

        for state_name, state_value in observation.get("state", {}).items():
            filename = f"state_{_safe_filename(str(state_name))}.png"
            if _save_state_plot(request_dir / filename, state_value):
                summary["files"].append(filename)

        with open(request_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._save_get_action_input(observation, options)
        return self.policy.get_action(observation, options)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return getattr(self.policy, "get_modality_config", lambda: {})()

    def check_observation(self, observation: dict[str, Any]) -> None:
        return self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        return self.policy.check_action(action)


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    save_client_inputs_dir: str | None = None
    """Directory to save visualized get_action inputs from clients. Disabled when unset."""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    if config.save_client_inputs_dir is not None:
        print(f"  Saving get_action input visualizations to: {config.save_client_inputs_dir}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                with open(config.modality_config_path, "r") as f:
                    raw = json.load(f)
                # ReplayPolicy expects ModalityConfig instances, not raw dicts.
                modality_configs = {k: ModalityConfig(**v) for k, v in raw.items()}
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    if config.save_client_inputs_dir is not None:
        policy = GetActionInputVisualizationPolicy(policy, config.save_client_inputs_dir)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    print(f"\n✓ Server ready — listening on {config.host}:{config.port}\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
