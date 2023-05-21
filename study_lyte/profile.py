from dataclasses. import dataclass
from enum import Enum
from . io import read_csv
from .adjustments import get_neutral_bias_at_border
from .detect import get_acceleration_start, get_acceleration_stop


@dataclass
class Event:
    name: str
    raw_index: int
    cropped_index: int
    depth: float # centimeters

class Sensor:
    UNAVAILABLE = -1



class LyteProfile:
    def __init__(self, filename, surface_detection_offset=4.5):
        """
        """
        self.filename = filename
        self.surface_detection_offset = surface_detection_offset

        # Properties
        self._df = None
        self._meta = None

        # Helpers
        self._motion_detect_column = None
        self._acceleration = None

        # Events
        self._start = None
        self._stop = None
        self._surface = None

    @property
    def df(self):
        if self._df is None:
            self._df, self._meta = read_csv(self.filename)
        return self._df

    @property
    def metadata(self):
        if self._df is None:
            self._df, self._meta = read_csv(self.filename)
        return self._meta

    @property
    def acceleration(self):
        if self._motion_detect_column is None:
            self._motion_detect_column = self.get_accelerometer_column(self.df.columns) or Sensor.UNAVAILABLE

        if self._acceleration is None and self._motion_detect_column != Sensor.UNAVAILABLE:
            # Remove gravity
            self._acceleration = get_neutral_bias_at_border(self.df[self._motion_detect_column])
        else:
            self._acceleration = Sensor.UNAVAILABLE
        return self._acceleration


    @property
    def start(self):
        """ Return start event """
        if self._start is not None:
            self._start = get_acceleration_start(self.acceleration)
        return self._start

    @property
    def stop(self):
        """ Return stop event """
        if self._stop is not None:
            self._stop = get_acceleration_stop(self.acceleration)
        return self._stop

    @property
    def surface(self):
        """ Return start event """
        if self._start is not None:
        return

    @staticmethod
    def get_accelerometer_column(columns):
        """
        Find a colum called acceleration or Y-Axis to handle older files
        """
        candidates = [c for c in columns if c.lower() in ['acceleration', 'y-axis']]
        if candidates:
            return candidates[0]
        else:
            return None
