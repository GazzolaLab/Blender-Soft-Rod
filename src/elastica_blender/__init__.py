__all__ = ["BlenderRodCallback"]

try:
    from .rod_callback import BlenderRodCallback
except ModuleNotFoundError:  # pragma: no cover
    pass
