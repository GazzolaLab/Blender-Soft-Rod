from typing import Protocol


class BlenderKeyframeManipulateProtocol(Protocol):
    def clear_animation(self) -> None: ...

    def set_keyframe(self, keyframe: int) -> None: ...
