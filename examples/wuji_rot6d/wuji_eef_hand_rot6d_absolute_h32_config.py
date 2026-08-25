"""GR00T modality config for Wuji/Astribot absolute EEF + hand actions (32 steps)."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


wuji_eef_hand_rot6d_absolute_h32_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_view",
            "left_wrist_view",
            "right_wrist_view",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_eef",
            "right_eef",
            "left_hand_joints",
            "right_hand_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=[
            "left_eef",
            "right_eef",
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(
    wuji_eef_hand_rot6d_absolute_h32_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
