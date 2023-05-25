from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from . io import read_csv
from .adjustments import get_neutral_bias_at_border, remove_ambient, apply_calibration
from .detect import get_acceleration_start, get_acceleration_stop, get_nir_surface
from .depth import get_depth_from_acceleration

@dataclass
class Event:
    name: str
    index: int
    depth: float # centimeters
    time: float # seconds

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
            self.calibration = calibration or {}

            # Properties
            self._df = None
            self._meta = None

            self._acceleration = None # No gravity acceleration
            self._cropped = None  # Full dataframe cropped to surface and stop
            self._force = None
            self._nir = None

            # Useful stats/info
            self._distance_traveled = None # distance travelled while moving
            self._distance_through_snow = None # Distance travelled while in snow
            self._motion_detect_column = None # column name containing accel data
            self._moving_time = None # time the probe was moving
            self._avg_velocity = None # avg velocity of the probe while in the snow

            self._datetime = None
            # Events
            self._start = None
            self._stop = None
            self._surface = None

        @property
        def raw(self):
            """
            Pandas dataframe hold the data exactly as it read in.
            """
            if self._df is None:
                self._df, self._meta = read_csv(str(self.filename))
                self._df = self._df.rename(columns={'depth':'filtereddepth'})
            return self._df


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
                    self.get_accelerometer_column(self.raw.columns) or \
                Sensor.UNAVAILABLE

            if self._acceleration is None:
                if self._motion_detect_column != Sensor.UNAVAILABLE:
                    # Remove gravity
                    self._acceleration = get_neutral_bias_at_border(self.raw[self._motion_detect_column])
                else:
                    self._acceleration = Sensor.UNAVAILABLE
            return self._acceleration

        @property
        def nir(self):
            """
            Retrieve the Active NIR sensor with ambient NIR removed
            """
            if self._nir is None:
                nir = remove_ambient(self.raw['Sensor3'], self.raw['Sensor2'])
                self._nir = pd.DataFrame({'nir': nir, 'depth': self.depth})
                self._nir = self._nir.iloc[self.surface.nir.index:self.stop.index].reset_index()
                self._nir = self._nir.drop(columns='index')
                self._nir['depth'] = self._nir['depth'] - self._nir['depth'].iloc[0]
            return self._nir

        @property
        def force(self):
            """
            calibrated force and depth as a pandas dataframe cropped to the snow surface and the stop of motion
            """
            if self._force is None:
                if 'Sensor1' in self.calibration.keys():
                    force = apply_calibration(self.raw['Sensor1'].values, self.calibration['Sensor1'])
                else:
                    force = self.raw['Sensor1'].values

                self._force = pd.DataFrame({'force': force, 'depth': self.depth.values})
                self._force = self._force.iloc[self.surface.force.index:self.stop.index].reset_index()
                self._force = self._force.drop(columns='index')
                self._force['depth'] = self._force['depth'] - self._force['depth'].iloc[0]

            return self._force

        @property
        def depth(self):
            if 'depth' not in self.raw.columns:
                df = pd.DataFrame.from_dict({'time':self.raw['time'],
                                             'acceleration':self.acceleration})
                depth = get_depth_from_acceleration(df).reset_index()
                self.raw['depth'] = depth[self._motion_detect_column]
            return self.raw['depth']

        @property
        def start(self):
            """ Return start event """
            if self._start is None:
                idx = get_acceleration_start(self.acceleration)
                depth = self.depth.iloc[idx]
                self._start = Event(name='start', index=idx, depth=depth, time=self.raw['time'].iloc[idx])
            return self._start

        @property
        def stop(self):
            """ Return stop event """
            if self._stop is None:
                idx = get_acceleration_stop(self.acceleration)
                depth = self.depth.iloc[idx]
                self._stop = Event(name='stop', index=idx, depth=depth, time=self.raw['time'].iloc[idx])
            return self._stop

        @property
        def surface(self):
            """
            Return surface events for the nir and force which are physically separated by a distance
            """
            if self._surface is None:
                idx = get_nir_surface(self.nir)
                depth = self.depth.iloc[idx]
                # Event according the NIR sensors
                nir = Event(name='surface', index=idx, depth=depth, time=self.raw['time'].iloc[idx])


                # Event according to the force sensor
                force_surface_depth = depth + self.surface_detection_offset
                f_idx = np.abs(self.depth - force_surface_depth).argmin()
                force = Event(name='surface', index=f_idx, depth=force_surface_depth, time=self.raw['time'].iloc[idx])
                self._surface = SimpleNamespace(name='surface', nir=nir, force=force)
            return self._surface

        @property
        def distance_traveled(self):
            if self._distance_traveled is None:
                self._distance_traveled = abs(self.start.depth - self.stop.depth)
            return self._distance_traveled

        @property
        def distance_through_snow(self):
            if self._distance_through_snow is None:
                self._distance_through_snow = abs(self.surface.nir.depth - self.stop.depth)
            return self._distance_through_snow

        @property
        def moving_time(self):
            """Amount of time the probe was in motion"""
            if self._moving_time is None:
                self._moving_time = self.stop.time - self.start.time
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
            return [self.start, self.stop, self.surface.nir, self.surface.force]

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
            profile_string += msg.format('Points', f'{len(self.raw.index):,}')
            profile_string += msg.format('Depth', f'{self.distance_traveled:0.1f} cm')
            profile_string += '-' * len(header) + '\n'
            return profile_string
