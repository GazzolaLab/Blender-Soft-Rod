import sys
from importlib import metadata as importlib_metadata

from .collections import *
from .rod import *
from .macros import *


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
