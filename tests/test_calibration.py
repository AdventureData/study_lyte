import pytest
from os.path import join
from pathlib import Path
from study_lyte.calibrations import Calibrations, MissingMeasurementDateException
import pandas as pd


class TestCalibrations:
    @pytest.fixture(scope='function')
    def calibration_json(self, data_dir):
        return Path(join(data_dir, 'calibrations.json'))

    @pytest.fixture(scope='function')
    def calibrations(self, calibration_json):
        return Calibrations(calibration_json)

    @pytest.mark.parametrize("serial, expected", [
        # Test actual calibration
        ("252813070A020004", -10),

        # Test no calibration found
        ("NONSENSE", -1),
    ])
    def test_attributes(self, calibrations, serial, expected):
        """"""
        result = calibrations.from_serial(serial)
        assert result.calibration['Sensor1'][2] == expected

    @pytest.mark.parametrize("serial, date, expected", [
        # Test valid multi calibration between date 1 and date 2
        ("252813070A020005", "2024-02-01", 200),
        # Test valid multi calibration with exact match on date 2
        ("252813070A020005", "2025-05-01", 600),
        # Test single calibration with a date provided.
        ("252813070A020004", "2025-01-01", 409),
    ])
    def test_date_based(self, calibrations, serial, date, expected):
        """"""
        dt = pd.to_datetime(date)
        result = calibrations.from_serial(serial, date=dt)
        assert result.calibration['Sensor1'][3] == expected

    def test_missing_measurement_date_exception(self, calibrations):
        """ Confirm this raises an exception when no date is provided """
        with pytest.raises(MissingMeasurementDateException):
            calibrations.from_serial("252813070A020005", date=None)
