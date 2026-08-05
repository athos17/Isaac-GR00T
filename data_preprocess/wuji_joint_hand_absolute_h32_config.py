"""GR00T modality config for Wuji absolute arm-joint + hand-joint actions."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


JOINT_MODALITY_KEYS = [
    "left_joint_space",
    "right_joint_space",
    "left_hand_joints",
    "right_hand_joints",
]

wuji_joint_hand_absolute_h32_config = {
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
        modality_keys=JOINT_MODALITY_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=JOINT_MODALITY_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in JOINT_MODALITY_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(
    wuji_joint_hand_absolute_h32_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
