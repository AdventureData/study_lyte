from enum import Enum
import json
from pathlib import Path
from .logging import setup_log
import logging
from dataclasses import dataclass
from typing import List
from datetime import datetime
import pandas as pd

setup_log()

LOG = logging.getLogger('study_lyte.calibrations')

class MissingMeasurementDateException(Exception):
    """
    Exception to raise when a probe has multiple calibrations but the date has 
    not been specified.
    """
    pass


@dataclass()
class Calibration:
    """Small class to make accessing calibration data a bit more convenient"""
    serial: str
    calibration: dict[str, List[float]]
    date: datetime = None


class Calibrations:
    """
    Class to read in a json containing calibrations, keyed by serial number and
     valued by dictionary of sensor names containing cal values
    """
    def __init__(self, filename:Path):
        with open(filename, mode='r') as fp:
            self._info = json.load(fp)

    def from_serial(self, serial:str, date: datetime=None) -> Calibration:
        """ Build data object from the calibration result """
        calibrations = self._info.get(serial)
        cal = None

        if calibrations is None:
            cal = self._info['default']
            serial = 'UNKNOWN'

        else:
            # Single calibration, returned as a dict
            if isinstance(calibrations, dict):
                cal = calibrations

            # Account for multiple calibrations
            elif isinstance(calibrations, list):
                # Check the date is provided
                if date is None and len(calibrations) > 1:
                    raise MissingMeasurementDateException("Multiple calibrations found, but no date provided")
                else:
                    # Find the calibration that matches the date
                    for c in calibrations:
                        if date >= pd.to_datetime(c['date']):
                            cal = c

                    # No matches were found, date is too early
                    if cal is None:
                        LOG.warning(f"All available calibrations for {serial} are not available before {date}, using default")
                        cal = self._info['default']
                        serial = 'UNKNOWN'

        if cal is not None and serial != 'UNKNOWN':
            LOG.info(f"Calibration found ({serial})!")
        else:
            LOG.warning(f"No calibration found for {serial}, using default")

        result = Calibration(serial=serial, calibration=cal)
        return result
