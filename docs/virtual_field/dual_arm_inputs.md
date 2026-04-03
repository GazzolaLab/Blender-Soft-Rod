# Interacting with Controller Data

This page covers how XR controller data moves through the runtime and how your mode can react to it.

## Customize Controller Command Mapping

You can customize controller behavior by overriding `handle_commands()`.
The base class already implements the default dual-arm behavior, so most modes should extend it rather than replace it entirely.

```python
    @override
    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        """
        Handle controller commands for a given arm.
        """
        super().handle_commands(
            arm_id,
            controller_command,
            previous_controller_command=previous_controller_command,
        )

        if bool(controller_command.buttons.get("grip_click", False)):
            ...

    @override
    def handle_command_inactive(self, arm_id: str) -> None:
        """
        Reset any per-frame controller-driven state when no input arrives.
        """
        super().handle_command_inactive(arm_id)
```

[ArmCommands](/api/virtual_field/commands.rst)

Default behavior is
- detach the arm by calling `set_attached()`, when the `primary` or `secondary` button is pressed.
- on the rising edge of `secondary`, reset the target pose to the rest pose by calling `reset_target_to_rest()`
- on the rising edge of `secondary`, recalibrate the controller orientation by calling `recalibrate_orientation_to_base()`
- otherwise forward the current controller pose to `set_target_pose()`

The base class derives the `secondary` rising edge from `previous_controller_command`, so your mode gets both the current input and the prior frame's input when needed.

You can find an example of using `trigger_click` and `grip_click` in `Cathy-Throw` mode.

## (Frontend) Controller data flow

Controller input enters the system as `XRInputSample`:

```python
XRInputSample(
    timestamp=...,
    head_pose=...,
    controllers={
        "left": ControllerSample(...),
        "right": ControllerSample(...),
    },
)
```


Each `ControllerSample` carries:

- `pose` with translation and quaternion rotation
- `velocity`
- analog `grip` and `trigger`
- 2D `joystick`
- boolean `buttons`

`SessionArmControlMapper` converts those into one `ArmCommand` per controlled arm. The important mapping is:

- `controller.pose` becomes `command.target`
- `controller.grip >= clutch_threshold` becomes `command.active`
- `controller.joystick` is deadbanded and copied through
- `controller.buttons` is passed through unchanged

## How the backend calls your mode

`MultiArmPassThroughBackend._apply_command()` is the bridge from controller commands into your simulation.

For simulation-backed modes it currently does the following:

- looks up the previous `ArmCommand` for that arm, if there is one
- calls `handle_commands(arm_id, controller_command, previous_controller_command=...)`
- stores the current command for the next frame
- if an arm does not receive a command in the current frame, calls `handle_command_inactive(arm_id)` and clears the stored previous command

That means the backend no longer needs to know mode-specific button mappings. Mode-specific controller behavior should live in your simulation class by overriding `handle_commands()` and, when needed, `handle_command_inactive()`.
