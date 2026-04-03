# Interacting with Controller Data

This page covers how XR controller data moves through the runtime and how your mode can react to it.

## Customize Controller Command Mapping

Simply, you can customize the controller command by overriding the `handle_commands` method.
This is default implementation in base class `DualArmSimulationBase`.

```python
    def handle_commands(self, arm_id: str, controller_command: ArmCommand) -> None:
        """
        Handle controller commands for a given arm.
        """
        if arm_id == self.arm_ids[0]:
            self.handle_commands_left(controller_command)
        elif arm_id == self.arm_ids[1]:
            self.handle_commands_right(controller_command)
        else:
            # TODO: Handle invalid arm ID.
            return None
```

[ArmCommands](/api/virtual_field/commands.rst)

If you want to separately handle the left and right arms, you can override the `handle_commands_left` and `handle_commands_right` methods.

```python
    def handle_commands_left(self, controller_command: ArmCommand) -> None:
        ...

    def handle_commands_right(self, controller_command: ArmCommand) -> None:
        ...
```

Default behavior is
- detach the arm by calling `set_attached()`, when the `primary` or `secondary` button is pressed.
- reset the target pose to the rest pose by calling `reset_target_to_rest()`, when the `secondary` button is pressed.

You can find example of using trigger and grip buttons in `Cathy-Throw` mode.

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

- `grip_click` calls `set_base_pull_active(arm_id, pressed)`
- `trigger_click` calls `set_sucker_active(arm_id, pressed)`
- `primary` or `secondary` detach the arm by calling `set_attached()`
- the rising edge of `secondary` triggers:

  - `reset_target_to_rest(arm_id)`
  - `recalibrate_orientation_to_base(arm_id, command.target.rotation_xyzw)`

- otherwise the current pose is forwarded to `set_target_pose(arm_id, translation, rotation_xyzw)`

That means the easiest way to react to controller data is to override the small hook methods that already exist on the base class.
