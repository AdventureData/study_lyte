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
            self._raw = None
            self._meta = None

            self._acceleration = None # No gravity acceleration
            self._cropped = None  # Full dataframe cropped to surface and stop
            self._force = None
            self._nir = None

            # Useful stats/info
            self._distance_traveled = None # distance travelled while moving
            self._distance_through_snow = None # Distance travelled while in snow
            self._motion_detect_name = None # column name containing accel data dir of travel
            self._acceleration_names = None # All columns containing accel data
            self._moving_time = None # time the probe was moving
            self._avg_velocity = None # avg velocity of the probe while in the snow
            self._resolution = None # Vertical resolution of the profile in the snow
            self._datetime = None
            # Events
            self._start = None
            self._stop = None
            self._surface = None

        @staticmethod
        def process_df(df):
            """
            Migrate all baro depths to filtereddepth and remove ambient
            to add NIR column
            """
            df = df.rename(columns={'depth': 'filtereddepth'})
            df['nir'] = remove_ambient(df['Sensor3'], df['Sensor2'])
            return df

        @classmethod
        def from_dataframe(cls, df):
            profile = LyteProfileV6(None)
            profile._raw = self.process_df(df)
            return profile

        @property
        def raw(self):
            """
            Pandas dataframe hold the data exactly as it read in.
            """
            if self._raw is None:
                self._raw, self._meta = read_csv(str(self.filename))
                self._raw = self.process_df(self.raw)

            return self._raw


        @property
        def metadata(self):
            """
            Returns a dictionary of all data held in the header portion of the csv
            """
            if self._raw is None:
                self._raw, self._meta = read_csv(str(self.filename))
                self._raw = self.process_df(self.raw)

            return self._meta

        @property
        def motion_detect_name(self):
            """Return all the names of acceleration columns"""
            if self._motion_detect_name is None:
                self._motion_detect_name = self.get_motion_name(self.raw.columns)
            return self._motion_detect_name

        @property
        def acceleration_names(self):
            """Return all the names of acceleration columns"""
            if self._acceleration_names is None:
                self._acceleration_names = self.get_acceleration_columns(self.raw.columns)
            return self._acceleration_names

        @property
        def acceleration(self):
            """
            Retrieve acceleration with gravity removed
            """
            # Assign the detection column if it is available
            if self._acceleration is None:
                if self.motion_detect_name != Sensor.UNAVAILABLE:
                    # Remove gravity
                    self._acceleration = get_neutral_bias_at_border(self.raw[self.motion_detect_name])
                else:
                    self._acceleration = Sensor.UNAVAILABLE
            return self._acceleration

        @property
        def nir(self):
            """
            Retrieve the Active NIR sensor with ambient NIR removed
            """
            if self._nir is None:
                self._nir = pd.DataFrame({'nir': self.raw['nir'], 'depth': self.depth})
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
                                             self.motion_detect_name:self.acceleration})
                depth = get_depth_from_acceleration(df).reset_index()
                self.raw['depth'] = depth[self.motion_detect_name]
            return self.raw['depth']

        @property
        def time(self):
            """Return the sample time data"""
            return self.raw['time']

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
                backward_accel = get_neutral_bias_at_border(self.raw[self.motion_detect_name], direction='backward')
                idx = get_acceleration_stop(backward_accel)
                depth = self.depth.iloc[idx]
                self._stop = Event(name='stop', index=idx, depth=depth, time=self.raw['time'].iloc[idx])
            return self._stop

        @property
        def surface(self):
            """
            Return surface events for the nir and force which are physically separated by a distance
            """
            if self._surface is None:
                # Call to populate nir in raw
                idx = get_nir_surface(self.raw['nir'])
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
        def resolution(self):
            if self._resolution is None:
                n_points = len(self.nir)
                self._resolution = n_points / self.distance_through_snow
            return self._resolution

        @property
        def events(self):
            """
            Return all the common events recorded
            """
            return [self.start, self.stop, self.surface.nir, self.surface.force]

        @staticmethod
        def get_motion_name(columns):
            """
            Find a column containing acceleration data sometimes called
            acceleration or Y-Axis to handle variations in formatting of file
            """
            candidates = [c for c in columns if c.lower() in ['acceleration', 'y-axis']]
            if candidates:
                return candidates[0]
            else:
                return Sensor.UNAVAILABLE

        @staticmethod
        def get_acceleration_columns(columns):
            """
            Find a columns containing acceleration data sometimes called
            acceleration or X,Y,Z-Axis to handle variations in formatting of file
            """
            candidates = [c for c in ['acceleration', 'X-Axis', 'Y-Axis', 'Z-Axis'] if c in columns]
            if candidates:
                return candidates
            else:
                return Sensor.UNAVAILABLE

        def report_card(self):
            """
            Return a useful string to print about metrics
            """
            msg = '| {:<15} {:<20} |\n'
            n_chars = int((39 - len(self.filename.name)) / 2)
            s =  '-'* n_chars
            header = f'\n{s} {self.filename.name} {s}\n'
            profile_string = header
            profile_string += msg.format('Recorded', f'{self.datetime.isoformat()}')
            profile_string += msg.format('Points', f'{len(self.raw.index):,}')
            profile_string += msg.format('Moving Time', f'{self.moving_time:0.1f} s')
            profile_string += msg.format('Avg. Speed', f'{self.avg_velocity:0.0f} cm/s')
            profile_string += msg.format('Resolution', f'{self.resolution:0.1f} pts/cm')
            profile_string += msg.format('Total Travel', f'{self.distance_traveled:0.1f} cm')
            profile_string += msg.format('Snow Depth', f'{self.distance_through_snow:0.1f} cm')
            profile_string += '-' * (len(header)-2) + '\n'
            return profile_string


        def __repr__(self):
            profile_str = f"LyteProfile (Recorded {len(self.raw):,} points, {self.datetime.isoformat()})"
            return profile_str
