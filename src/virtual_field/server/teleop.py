from __future__ import annotations

from virtual_field.core.commands import MultiArmCommand, XRInputSample
from virtual_field.core.interfaces import ControlMapper


class TeleopService:
    def __init__(self, mapper: ControlMapper) -> None:
        self.mapper = mapper

    def map_input(self, sample: XRInputSample) -> MultiArmCommand:
        return self.mapper.map_input(sample)
