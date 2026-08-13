import numpy as np
import pytest
from robot_data_pipeline.processing.rotations import (
    RotationError,
    make_quaternion_signs_continuous,
    quaternion_to_matrix,
    quaternion_to_rot6d,
    slerp,
)


def test_quaternion_sign_does_not_change_rot6d() -> None:
    quaternion = np.array([0.2, -0.3, 0.1, 0.9])

    assert np.allclose(quaternion_to_rot6d(quaternion), quaternion_to_rot6d(-quaternion))


def test_rot6d_uses_first_two_rotation_matrix_rows() -> None:
    result = quaternion_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0]))

    assert np.allclose(result, [1, 0, 0, 0, 1, 0])


def test_sign_continuity_flips_equivalent_neighbor() -> None:
    quaternions = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])

    result = make_quaternion_signs_continuous(quaternions)

    assert np.dot(result[0], result[1]) > 0


def test_slerp_halfway_between_identity_and_90_degree_z_rotation() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    matrix = quaternion_to_matrix(slerp(q0, q1, 0.5))
    expected = np.array(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(matrix, expected)


def test_zero_norm_quaternion_is_rejected() -> None:
    with pytest.raises(RotationError, match="norm outside"):
        quaternion_to_rot6d(np.zeros(4))
