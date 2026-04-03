# Common Virtual Field Utilities

(target-pose-proportional-control)=
## `TargetPoseProportionalControl`

`TargetPoseProportionalControl` is the main utility force used by the current
dual-arm simulation modes to make a rod tip follow a controller-driven target
pose.

Source:
{doc}`api/runtime_common_utilities`

### Expected target orientation convention

The controller target orientation is expected in the same row-wise director
format used elsewhere in the runtime:

- rows represent `[normal, binormal, tangent]` in world coordinates
- the matrix maps world coordinates into local rod coordinates

This is why the runtime converts controller quaternions before passing the
orientation into the control force.

### Notes

- If `is_attached()` returns `False`, the control applies nothing.
- The force is proportional only; there is no derivative term in this class.
- Angular control is computed from the orientation error between the current
  director frame and the target director frame.
- The implementation includes a special-case path for near-180-degree rotation
  differences to avoid unstable axis recovery.
