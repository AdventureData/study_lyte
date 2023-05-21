from dataclasses import dataclass
from enum import Enum
import pandas as pd

from . io import read_csv
from .adjustments import get_neutral_bias_at_border, remove_ambient
from .detect import get_acceleration_start, get_acceleration_stop, get_nir_surface
from .depth import get_depth_from_acceleration

@dataclass
class Event:
    name: str
    index: int
    depth: float # centimeters


class Sensor(Enum):
    """Enum for various scenarios that come up with variations of data"""
    UNAVAILABLE = -1


class LyteProfileV6:
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
    def raw_df(self):
        if self._df is None:
            self._df, self._meta = read_csv(self.filename)
            self._df = self._df.rename(columns={'depth':'filtereddepth'})
        return self._df

    def cropped(self):
        return self.raw_df.iloc[self.surface.index:self.stop.index]

    @property
    def metadata(self):
        if self._df is None:
            self._df, self._meta = read_csv(self.filename)
        return self._meta

    @property
    def acceleration(self):
        if self._motion_detect_column is None:
            self._motion_detect_column = self.get_accelerometer_column(self.raw_df.columns) or Sensor.UNAVAILABLE

        if self._acceleration is None:
            if self._motion_detect_column != Sensor.UNAVAILABLE:
                # Remove gravity
                self._acceleration = get_neutral_bias_at_border(self.raw_df[self._motion_detect_column])
            else:
                self._acceleration = Sensor.UNAVAILABLE
        return self._acceleration

    @property
    def nir(self):
        if 'nir' not in self.raw_df.columns:
            nir = remove_ambient(self.raw_df['Sensor3'], self.raw_df['Sensor2'])
            self.raw_df['nir'] = nir
        return self.raw_df['nir']
    
    @property
    def depth(self):
        if 'depth' not in self.raw_df.columns:
            df = pd.DataFrame.from_dict({'time':self.raw_df['time'], 'acceleration':self.acceleration})
            depth = get_depth_from_acceleration(df).reset_index()
            self.raw_df['depth'] = depth[self._motion_detect_column]
        return self.raw_df['depth']

    @property
    def start(self):
        """ Return start event """
        if self._start is None:
            idx = get_acceleration_start(self.acceleration)
            depth = self.depth.iloc[idx]
            self._start = Event(name='start', index=idx, depth=depth)
        return self._start

    @property
    def stop(self):
        """ Return stop event """
        if self._stop is None:
            idx = get_acceleration_stop(self.acceleration)
            depth = self.depth.iloc[idx]
            self._stop = Event(name='stop', index=idx, depth=depth)
        return self._stop

    @property
    def surface(self):
        """ Return start event """
        if self._surface is None:
            idx = get_nir_surface(self.nir)
            depth = self.depth.iloc[idx]
            self._surface = Event(name='surface', index=idx, depth=depth)
        return self._surface
    @property
    def events(self):
        return [self.start, self.stop, self.surface]

    @staticmethod
    def get_accelerometer_column(columns):
        """
        Find a column called acceleration or Y-Axis to handle variations in formatting
        of files
        """
        candidates = [c for c in columns if c.lower() in ['acceleration', 'y-axis']]
        if candidates:
            return candidates[0]
        else:
            return None
