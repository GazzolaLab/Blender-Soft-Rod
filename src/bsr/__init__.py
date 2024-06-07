import sys
from importlib import metadata as importlib_metadata

from .collections import *
from .file import *
from .macros import *
from .rod import *


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
