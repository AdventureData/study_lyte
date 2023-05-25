from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pathlib import Path

import pytest

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
    def __init__(self, filename, surface_detection_offset=4.5, calibration=None):
        """
        Args:
            filename: path to valid lyte probe csv.
            surface_detection_offset: Geometric offset between nir sensors and tip in cm.
            calibration: Dictionary of keys and polynomial coefficients to calibration sensors
        """
        self.filename = Path(filename)
        self.surface_detection_offset = surface_detection_offset

        # Properties
        self._df = None
        self._meta = None

        self._motion_detect_column = None # column name containing accel data
        self._acceleration = None # No gravity acceleration
        self._cropped = None  # Full dataframe cropped to surface and stop
        self._distance_travelled = None # distance travelled in snow
        self._moving_time = None # time the probe was moving
        self._avg_velocity = None # avg velocity of the probe while in the snow

        self._datetime = None
        # Events
        self._start = None
        self._stop = None
        self._surface = None

    @property
    def raw_df(self):
        """
        Pandas dataframe hold the data exactly as it read in.
        """
        if self._df is None:
            self._df, self._meta = read_csv(str(self.filename))
            self._df = self._df.rename(columns={'depth':'filtereddepth'})
        return self._df

    @property
    def cropped(self):
        """
        Provides the pandas dataframe cropped to the snow surface and the stop of motion
        """
        if self._cropped is None:
            # Crop and reset index so surface = index(0)
            self._cropped = self.raw_df.iloc[self.surface.index:self.stop.index].reset_index()
            self._cropped['depth'] = self._cropped['depth'] - self._cropped['depth'].iloc[0]
        return self._cropped

    @property
    def metadata(self):
        """
        Returns a dictionary of all data held in the header portion of the csv
        """
        if self._df is None:
            self._df, self._meta = read_csv(str(self.filename))
        return self._meta

    @property
    def acceleration(self):
        """
        Retrieve acceleration with gravity removed
        """
        # Assign the detection column if it is available
        if self._motion_detect_column is None:
            self._motion_detect_column = \
                self.get_accelerometer_column(self.raw_df.columns) or \
            Sensor.UNAVAILABLE

        if self._acceleration is None:
            if self._motion_detect_column != Sensor.UNAVAILABLE:
                # Remove gravity
                self._acceleration = get_neutral_bias_at_border(self.raw_df[self._motion_detect_column])
            else:
                self._acceleration = Sensor.UNAVAILABLE
        return self._acceleration

    @property
    def nir(self):
        """
        Retrieve the Active NIR sensor with ambient NIR removed
        """
        if 'nir' not in self.raw_df.columns:
            nir = remove_ambient(self.raw_df['Sensor3'], self.raw_df['Sensor2'])
            self.raw_df['nir'] = nir
        return self.raw_df['nir']
    
    @property
    def depth(self):
        if 'depth' not in self.raw_df.columns:
            df = pd.DataFrame.from_dict({'time':self.raw_df['time'],
                                         'acceleration':self.acceleration})
            depth = get_depth_from_acceleration(df).reset_index()
            self.raw_df['depth'] = depth[self._motion_detect_column].mul(100)
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
        """ Return surface event """
        if self._surface is None:
            idx = get_nir_surface(self.nir)
            depth = self.depth.iloc[idx]
            self._surface = Event(name='surface', index=idx, depth=depth)
        return self._surface

    @property
    def distance_traveled(self):
        if self._distance_travelled is None:
            self._distance_travelled = self.cropped['depth'].max() - self.cropped['depth'].min()
        return self._distance_travelled

    @property
    def moving_time(self):
        """Amount of time the probe was in motion"""
        if self._moving_time is None:
            self._moving_time = self.cropped['time'].max() - self.cropped['time'].min()
        return self._moving_time
    @property
    def avg_velocity(self):
        if self._avg_velocity is None:
            self._avg_velocity = self.distance_traveled / self.moving_time
        return self._avg_velocity

    @property
    def datetime(self):
        if self._datetime is None:
            self._datetime = pd.to_datetime(self.metadata['RECORDED'])
        return self._datetime

    @property
    def events(self):
        """
        Return all the common events recorded
        """
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

    def __repr__(self):
        msg = '\n| {:<10} {:<10} \n'
        header = f'+---- {self.filename.name} ----+\n'
        profile_string = header
        profile_string += msg.format('Recorded', f'{self.datetime.isoformat()}')
        profile_string += msg.format('Points', f'{len(self.raw_df.index):,}')
        profile_string += msg.format('Depth', f'{self.distance_traveled:0.1f} cm')
        profile_string += '-' * len(header) + '\n'
        return profile_string
