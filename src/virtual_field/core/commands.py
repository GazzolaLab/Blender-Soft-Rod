from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import Transform, Twist


JSONDict = dict[str, Any]


def _validate_size(values: list[float], size: int, name: str) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have size {size}, got {len(values)}")


@dataclass(slots=True)
class ArmCommand:
    """One controller input for a simulation step.

    Built from ``ControllerSample`` by ``SessionArmControlMapper``; the Virtual
    Field *Interacting with Controller Data* doc describes the mapping from XR
    samples to these fields.

    Attributes
    ----------
    arm_id
        Stable arm identifier; must match the corresponding key in
        ``MultiArmCommand.commands``.
    active
        Typically ``True`` when analog grip is at or above the session clutch
        threshold; used to gate pass-through updates. Simulation-backed modes may
        still read ``buttons`` every frame.
    target
        Controller pose as :class:`Transform` (same role as
        ``ControllerSample.pose``).
    velocity
        Optional twist from the controller (linear and angular); defaults to
        zeros if omitted on the wire.
    joystick
        Two floats after deadbanding (thumbstick axes), length validated to 2.
    buttons
        Named booleans passed through from the frontend. The dual-arm backend
        uses ``grip_click``, ``trigger_click``, ``primary``, and ``secondary``
        (rising edge on ``secondary`` resets/rests and recalibrates orientation).
    """

    arm_id: str
    active: bool
    target: Transform
    velocity: Twist = field(default_factory=Twist)
    joystick: list[float] = field(default_factory=lambda: [0.0, 0.0])
    buttons: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("arm_id cannot be empty")
        _validate_size(self.joystick, 2, "joystick")

    def to_dict(self) -> JSONDict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "arm_id": self.arm_id,
            "active": self.active,
            "target": self.target.to_dict(),
            "velocity": self.velocity.to_dict(),
            "joystick": self.joystick,
            "buttons": self.buttons,
        }


@dataclass(slots=True)
class MultiArmCommand:
    """One frame of controller commands for all currently tracked arms.

    This is the runtime-level command packet consumed by
    ``MultiArmPassThroughBackend.step()``. Each entry in ``commands`` is keyed
    by ``arm_id`` and must contain a matching :class:`ArmCommand`.

    Attributes
    ----------
    timestamp
        Sample time from the originating XR input frame.
    commands
        Mapping from ``arm_id`` to per-arm command data. In the dual-arm setup
        this usually contains zero, one, or two entries depending on which
        controllers are currently present in the XR sample.
    """

    timestamp: float
    commands: dict[str, ArmCommand]

    def __post_init__(self) -> None:
        for key, command in self.commands.items():
            if key != command.arm_id:
                raise ValueError("command keys must match ArmCommand.arm_id")

    def to_dict(self) -> JSONDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "commands": {
                arm_id: command.to_dict()
                for arm_id, command in self.commands.items()
            },
        }


@dataclass(slots=True)
class ControllerSample:
    """Raw XR controller sample before mapping into arm-space commands.

    ``SessionArmControlMapper`` consumes this data and produces an
    :class:`ArmCommand`. The *Interacting with Controller Data* guide describes
    the default mapping used by the virtual field runtime.

    Attributes
    ----------
    pose
        World-space controller pose as a :class:`Transform`.
    velocity
        Optional world-space linear and angular controller velocity.
    grip
        Analog grip value, typically in ``[0, 1]``. The default mapper compares
        this against the clutch threshold to determine ``ArmCommand.active``.
    trigger
        Analog trigger value, typically in ``[0, 1]``.
    joystick
        Two thumbstick axes in ``[x, y]`` order. The default mapper applies a
        deadband before copying these values into :class:`ArmCommand`.
    buttons
        Frontend-supplied boolean button state. Common keys include
        ``primary``, ``secondary``, ``grip_click``, and ``trigger_click``.
    """

    pose: Transform
    velocity: Twist = field(default_factory=Twist)
    grip: float = 0.0
    trigger: float = 0.0
    joystick: list[float] = field(default_factory=lambda: [0.0, 0.0])
    buttons: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_size(self.joystick, 2, "joystick")

    def to_dict(self) -> JSONDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "pose": self.pose.to_dict(),
            "velocity": self.velocity.to_dict(),
            "grip": self.grip,
            "trigger": self.trigger,
            "joystick": self.joystick,
            "buttons": self.buttons,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "ControllerSample":
        """Deserialize a controller sample from JSON-compatible data."""
        velocity_data = data.get("velocity", {})
        if velocity_data:
            velocity = Twist.from_dict(velocity_data)
        else:
            velocity = Twist()
        return cls(
            pose=Transform.from_dict(data["pose"]),
            velocity=velocity,
            grip=float(data.get("grip", 0.0)),
            trigger=float(data.get("trigger", 0.0)),
            joystick=list(data.get("joystick", [0.0, 0.0])),
            buttons=dict(data.get("buttons", {})),
        )


@dataclass(slots=True)
class XRInputSample:
    """One WebXR input frame containing head and controller state.

    This is the Python-side representation of the ``xr_input`` payload described
    in the communication and controller-input guides.

    Attributes
    ----------
    timestamp
        Client-provided sample time.
    head_pose
        World-space head pose for the XR frame.
    controllers
        Mapping from controller hand labels such as ``"left"`` and ``"right"``
        to :class:`ControllerSample` values.
    """

    timestamp: float
    head_pose: Transform
    controllers: dict[str, ControllerSample]

    def __post_init__(self) -> None:
        if not self.controllers:
            raise ValueError("controllers cannot be empty")

    @classmethod
    def from_dict(cls, data: JSONDict) -> "XRInputSample":
        """Deserialize an XR input sample from JSON-compatible data."""
        return cls(
            timestamp=float(data["timestamp"]),
            head_pose=Transform.from_dict(data["head_pose"]),
            controllers={
                hand: ControllerSample.from_dict(controller)
                for hand, controller in data["controllers"].items()
            },
        )

    def to_dict(self) -> JSONDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "head_pose": self.head_pose.to_dict(),
            "controllers": {
                hand: controller.to_dict()
                for hand, controller in self.controllers.items()
            },
        }
