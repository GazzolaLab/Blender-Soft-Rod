import pytest

from virtual_field.core.commands import ArmCommand
from virtual_field.core.state import MeshEntity, Transform
from virtual_field.server.backends import MultiArmPassThroughBackend

pytestmark = pytest.mark.modules


def test_register_user_rejects_unsupported_mode() -> None:
    backend = MultiArmPassThroughBackend()
    with pytest.raises(ValueError, match="Unsupported character mode"):
        backend.register_user("user_invalid", character_mode="unknown-mode")


def test_update_mesh_transform_rejects_owner_mismatch() -> None:
    backend = MultiArmPassThroughBackend()
    backend.add_or_update_mesh(
        mesh=MeshEntity(
            mesh_id="mesh_1",
            owner_id="owner_a",
            asset_uri="data:model/gltf-binary;base64,AA==",
        )
    )
    updated = backend.update_mesh_transform(
        mesh_id="mesh_1", owner_id="owner_b", translation=[1.0, 0.0, 0.0]
    )
    assert updated is False


def test_apply_command_pass_through_ignores_inactive() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user("user_two_cr", character_mode="two-cr")
    arm = backend._arms[arm_ids[0]]
    before = arm.tip.translation.copy()
    backend._simulations.pop("user_two_cr", None)
    backend._apply_command(
        arm,
        ArmCommand(
            arm_id=arm_ids[0],
            active=False,
            target=Transform(translation=[2.0, 2.0, -2.0], rotation_xyzw=[0, 0, 0, 1]),
            buttons={},
        ),
    )
    assert arm.tip.translation == before
