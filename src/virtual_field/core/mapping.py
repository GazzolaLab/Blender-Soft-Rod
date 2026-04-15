from __future__ import annotations

from dataclasses import dataclass, field

from .commands import ArmCommand, MultiArmCommand, XRInputSample
from .interfaces import ControlMapper
from .state import Transform


def _apply_deadband(value: float, deadband: float) -> float:
    if abs(value) < deadband:
        return 0.0
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class DualArmControlMapper(ControlMapper):
    left_arm_id: str = "left_arm"
    right_arm_id: str = "right_arm"
    clutch_threshold: float = 0.4
    joystick_deadband: float = 0.1
    max_translation: float = 2.0
    smoothing: float = 0.25

    def __post_init__(self) -> None:
        self._smoothed_translation: dict[str, list[float]] = {}

    def map_input(self, sample: XRInputSample) -> MultiArmCommand:
        controller_to_arm = {
            "left": self.left_arm_id,
            "right": self.right_arm_id,
        }

        commands: dict[str, ArmCommand] = {}
        for hand, arm_id in controller_to_arm.items():
            controller = sample.controllers.get(hand)
            if controller is None:
                continue

            target = self._smooth_target(arm_id, controller.pose)
            active = controller.grip >= self.clutch_threshold
            joystick = [
                _apply_deadband(controller.joystick[0], self.joystick_deadband),
                _apply_deadband(controller.joystick[1], self.joystick_deadband),
            ]
            commands[arm_id] = ArmCommand(
                arm_id=arm_id,
                active=active,
                target=target,
                velocity=controller.velocity,
                joystick=joystick,
                buttons=controller.buttons,
            )

        actions = dict(sample.actions)
        actions.setdefault(
            "crawl",
            any(
                bool(controller.buttons.get("trigger_click", False))
                for controller in sample.controllers.values()
            ),
        )
        return MultiArmCommand(
            timestamp=sample.timestamp,
            commands=commands,
            head_pose=sample.head_pose,
            actions=actions,
        )

    def _smooth_target(self, arm_id: str, target: Transform) -> Transform:
        prev = self._smoothed_translation.get(arm_id, target.translation)
        smoothed = [
            prev[idx] + self.smoothing * (target.translation[idx] - prev[idx])
            for idx in range(3)
        ]
        clamped = [
            _clamp(value, -self.max_translation, self.max_translation)
            for value in smoothed
        ]
        self._smoothed_translation[arm_id] = clamped
        return Transform(
            translation=clamped,
            rotation_xyzw=target.rotation_xyzw,
        )


@dataclass(slots=True)
class SessionArmControlMapper(ControlMapper):
    controlled_arm_ids: tuple[str, str]
    clutch_threshold: float = 0.4
    joystick_deadband: float = 0.1
    max_translation: float = 2.0
    smoothing: float = 0.25

    def __post_init__(self) -> None:
        self._smoothed_translation: dict[str, list[float]] = {}

    def map_input(self, sample: XRInputSample) -> MultiArmCommand:
        controller_to_arm = {
            "left": self.controlled_arm_ids[0],
            "right": self.controlled_arm_ids[1],
        }

        commands: dict[str, ArmCommand] = {}
        for hand, arm_id in controller_to_arm.items():
            controller = sample.controllers.get(hand)
            if controller is None:
                continue

            target = self._smooth_target(arm_id, controller.pose)
            active = controller.grip >= self.clutch_threshold
            joystick = [
                _apply_deadband(controller.joystick[0], self.joystick_deadband),
                _apply_deadband(controller.joystick[1], self.joystick_deadband),
            ]
            commands[arm_id] = ArmCommand(
                arm_id=arm_id,
                active=active,
                target=target,
                velocity=controller.velocity,
                joystick=joystick,
                buttons=controller.buttons,
            )

        actions = dict(sample.actions)
        actions.setdefault(
            "crawl",
            any(
                bool(controller.buttons.get("trigger_click", False))
                for controller in sample.controllers.values()
            ),
        )
        return MultiArmCommand(
            timestamp=sample.timestamp,
            commands=commands,
            head_pose=sample.head_pose,
            actions=actions,
        )

    def _smooth_target(self, arm_id: str, target: Transform) -> Transform:
        prev = self._smoothed_translation.get(arm_id, target.translation)
        smoothed = [
            prev[idx] + self.smoothing * (target.translation[idx] - prev[idx])
            for idx in range(3)
        ]
        clamped = [
            _clamp(value, -self.max_translation, self.max_translation)
            for value in smoothed
        ]
        self._smoothed_translation[arm_id] = clamped
        return Transform(
            translation=clamped,
            rotation_xyzw=target.rotation_xyzw,
        )
