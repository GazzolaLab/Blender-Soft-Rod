import pytest

from virtual_field.core.commands import (
    ArmCommand,
    ControllerDisconnectedError,
    ControllerSample,
    MultiArmCommand,
    XRInputSample,
)
from virtual_field.core.mapping import SessionArmControlMapper
from virtual_field.core.state import Transform

pytestmark = pytest.mark.modules


def test_arm_command_rejects_empty_arm_id() -> None:
    with pytest.raises(ControllerDisconnectedError, match="arm_id cannot be empty"):
        ArmCommand(
            arm_id="",
            active=True,
            target=Transform(),
            buttons={},
        )


def test_multi_arm_command_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="command keys must match"):
        MultiArmCommand(
            timestamp=0.0,
            commands={
                "left_arm": ArmCommand(
                    arm_id="right_arm",
                    active=True,
                    target=Transform(),
                    buttons={},
                )
            },
        )


def test_xr_input_rejects_empty_controllers() -> None:
    with pytest.raises(ControllerDisconnectedError, match="controllers cannot be empty"):
        XRInputSample(timestamp=0.0, head_pose=Transform(), controllers={})


def test_session_mapper_defaults_crawl_from_trigger_click() -> None:
    mapper = SessionArmControlMapper(controlled_arm_ids=("a0", "a1"))
    sample = XRInputSample(
        timestamp=1.0,
        head_pose=Transform(),
        controllers={
            "left": ControllerSample(
                pose=Transform(),
                joystick=[0.01, -0.01],
                buttons={"trigger_click": True},
            )
        },
    )
    command = mapper.map_input(sample)
    assert command.actions["crawl"] is True
    assert "a0" in command.commands
    assert command.commands["a0"].joystick == [0.0, 0.0]
