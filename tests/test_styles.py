"""
Place to store common styling of plots as well as streamlining common tasks
like labeling and coloring of events and sensors.
"""

from study_lyte.styles import EventStyle, SensorStyle
import pytest

class TestEventStyle:
    @pytest.mark.parametrize("name, expected", [
        ('error', EventStyle.ERROR),
        ('blarg', EventStyle.UNKNOWN)

    ])
    def test_from_name(self, name, expected):
        style = EventStyle.from_name(name)
        assert style == expected

    @pytest.mark.parametrize("property, expected",[
        ('color', 'g'),
        ('linestyle', '--'),
        ('linewidth', 1),
        ('label', 'Start'),
    ])
    def test_property(self, property, expected):
        assert getattr(EventStyle.START, property) == expected


class TestSensorStyle:
    @pytest.mark.parametrize("name, expected", [
        ('Sensor1', SensorStyle.RAW_FORCE),
    ])
    def test_from_column(self, name, expected):
        style = SensorStyle.from_column(name)
        assert style == expected

    @pytest.mark.parametrize("property, expected",[
        ('column', 'fused'),
        ('label', 'Fused'),
        ('color', 'magenta'),
    ])
    def test_property(self, property, expected):
        assert getattr(SensorStyle.FUSED, property) == expected