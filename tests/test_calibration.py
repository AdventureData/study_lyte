import pytest
from os.path import join
from pathlib import Path
from study_lyte.calibrations import Calibrations


class TestCalibrations:
    @pytest.fixture(scope='function')
    def calibration_json(self, data_dir):
        return Path(join(data_dir, 'calibrations.json'))

    @pytest.fixture(scope='function')
    def calibrations(self, calibration_json):
        return Calibrations(calibration_json)

    @pytest.mark.parametrize("serial, expected", [
        ("252813070A020004", -10),
        ("NONSENSE", -1),
    ])
    def test_attributes(self, calibrations, serial, expected):
        """"""
        result = calibrations.from_serial(serial)
        assert result.calibration['Sensor1'][2] == expected


