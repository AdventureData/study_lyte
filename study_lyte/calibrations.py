from enum import Enum
import json
from pathlib import Path
from .logging import setup_log
import logging
from dataclasses import dataclass
from typing import List
setup_log()

LOG = logging.getLogger('study_lyte.calibrations')


@dataclass()
class Calibration:
    """Small class to make accessing calibration data a bit more convenient"""
    serial: str
    calibration: dict[str, List[float]]


class Calibrations:
    """
    Class to read in a json containing calibrations, keyed by serial number and
     valued by dictionary of sensor names containing cal values
    """
    def __init__(self, filename:Path):
        with open(filename, mode='r') as fp:
            self._info = json.load(fp)

    def from_serial(self, serial:str) -> Calibration:
        """ Build data object from the calibration result """
        cal = self._info.get(serial)
        if cal is None:
            LOG.warning(f"No Calibration found for serial {serial}, using default")
            cal = self._info['default']
            serial = 'UNKNOWN'

        else:
            LOG.info(f"Calibration found ({serial})!")

        result = Calibration(serial=serial, calibration=cal)
        return result
