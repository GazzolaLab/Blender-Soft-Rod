from typing import Protocol


class BlenderKeyframeManipulateProtocol(Protocol):
    def clear_animation(self) -> None: ...

    def update_keyframe(self, keyframe: int) -> None: ...
