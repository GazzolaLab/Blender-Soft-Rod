from typing import cast

from importlib import import_module

__all__ = ["BlenderRodCallback"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("elastica_blender.rod_callback")
    value = getattr(module, name)
    return cast(object, value)
