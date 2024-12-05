import pandas as pd
from scipy.integrate import cumulative_trapezoid
import numpy as np
from types import SimpleNamespace

from .decorators import time_series
from .detect import nearest_peak
from .adjustments import zfilter

@time_series
def get_depth_from_acceleration(acceleration_df: pd.DataFrame) -> pd.DataFrame:
    """
    Double integrate the acceleration to calculate a depth profile
    Assumes a starting position and velocity of zero. Convert to cm
    and return the data

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's without gravity

    Return:
        position_df: pandas Dataframe containing the same input axes plus magnitude of the result position
    """
    if type(acceleration_df) != pd.DataFrame:
        acceleration_df = pd.DataFrame(acceleration_df)
        # Auto gather the x,y,z acceleration columns if they're there.
    acceleration_columns = [c for c in acceleration_df.columns if 'Axis' in c or 'acceleration' == c]

    # Convert from g's to m/s2
    g = -9.81
    acc = acceleration_df[acceleration_columns].mul(g)

    # Calculate position
    position_vec = {}
    for i, axis in enumerate(acceleration_columns):
        # Integrate acceleration to velocity
        v = cumulative_trapezoid(acc[axis].values, acc.index, initial=0)
        # Integrate velocity to position
        position_vec[axis] = cumulative_trapezoid(v, acc.index, initial=0)

    position_df = pd.DataFrame.from_dict(position_vec)
    position_df['time'] = acc.index
    position_df = position_df.set_index('time')

    # Calculate the magnitude if all the components are available
    if all([c in acceleration_columns for c in ['X-Axis', 'Y-Axis', 'Z-Axis']]):
        position_arr = np.array([position_vec['X-Axis'],
                                 position_vec['Y-Axis'],
                                 position_vec['Z-Axis']])
        position_df['magnitude'] = np.linalg.norm(position_arr, axis=0)
    return position_df.mul(100)


@time_series
def get_fitted_depth(df: pd.DataFrame, column='depth', poly_deg=5) -> pd.DataFrame:
    """
    Fits a polynomial to the relative depth data specified and returns the
    fitted data.

    Args:
        df: pd.DataFrame containing
        column: Column to fit a polynomial to
        poly_deg: Integer of the polynomial degree to use

    Returns:
        fitted: pd.Dataframe indexed by time containing a new column named by the name of the column used
                but with fitted_ prepended e.g. fitted_depth
    """
    fitted = df[[column]].copy()
    coef = np.polyfit(fitted.index, fitted[column].values, deg=poly_deg)
    poly = np.poly1d(coef)
    df[f'fitted_{column}'] = poly(df.index)
    return df

@time_series
def get_constrained_baro_depth(baro_depth, start, stop, method='nanmedian'):
    """
    The Barometer depth is often stretched in time. Use the start and stop of the
    Accelerometer to constrain the peak/valley of the barometer, then rescale
    it by the tails.
    Args:
        baro_depth: Pandas series of barometer calculated depth indexed by time
        start: Index of start of motion to constrain the barometer
        stop: Index of stop of motion to constrain barometer
        method: aggregating method applied to data before the start and after stop
    """
    window_func = getattr(np, method)
    mid = int((stop + start) / 2)
    n_points = len(baro_depth)
    top_search = baro_depth.iloc[:mid]
    default_top = np.where(top_search == top_search.max())[0][0]
    top = nearest_peak(baro_depth.values, start, default_index=default_top, height=-10, distance=100)

    # Find valleys after, select closest to midpoint
    soft_stop = mid + int(0.1 * n_points)
    if soft_stop > len(baro_depth.index):
        soft_stop = len(baro_depth.index) - 1

    valley_search = baro_depth.iloc[mid:].values
    v_min = np.nanmin(valley_search)
    vmin_idx = np.where(valley_search == v_min)[0][0]
    bottom = nearest_peak(-1 * valley_search, stop - mid, default_index=vmin_idx, height=-10, distance=100)
    bottom += mid

    if bottom == stop:
        bot_mean_idx = bottom

    elif bottom >= n_points - 1:
        bot_mean_idx = n_points - 1

    else:
        bot_mean_idx = bottom - 1
    # Rescale
    top_mean = window_func(baro_depth.iloc[:top + 1])
    bottom_mean = window_func(baro_depth.iloc[bot_mean_idx:])
    delta_new = top_mean - bottom_mean
    delta_old = baro_depth.iloc[top] - baro_depth.iloc[bottom]

    depth_values = baro_depth.iloc[top:bottom + 1].values
    baro_time = np.linspace(baro_depth.index[start], baro_depth.index[stop], len(depth_values))
    result = pd.DataFrame.from_dict({'baro': depth_values, 'time': baro_time})
    result['baro'] = (result['baro'] - baro_depth.iloc[bottom]).div(delta_old).mul(delta_new)

    constrained = result.set_index('time')
    #assume_no_upward_motion(result[baro])
    constrained = constrained - constrained.iloc[0]
    return constrained


class DepthTimeseries:
    """
    Class for managing depth time series data
    """
    def __init__(self, series, start_idx=None, stop_idx=None, origin=None):
        # Hang on to the raw data
        self.raw = series

        # Keep track of the start stop
        self.start_idx = start_idx
        self.stop_idx = stop_idx

        # Establish a zero depth index
        self.origin = origin
        if self.origin is None:
            self.origin = start_idx


        # Holder for depth data with at least the origin zeroed out
        self._depth = None

        # Useful attributes
        self._avg_velocity = None
        self._distance_traveled = None
        self._distance_traveled_during_motion = None
        self._avg_distance_traveled = None

        self._has_upward_motion = None
        self._velocity = None
        self._velocity_range = None
        self._max_velocity = None

    @property
    def depth(self):
        if self._depth is None:
            self._depth = self.raw - self.raw.iloc[self.origin]
        return self._depth

    @property
    def velocity(self):
        if self._velocity is None:
            dt = self.depth.index[1] - self.depth.index[0]
            velocity = np.gradient(self.depth, dt)
            # Due to rounding issues the index is not evenly space, so filter the velocity
            velocity = zfilter(velocity, 0.01)
            self._velocity = pd.Series(velocity, self.depth.index, name='velocity')
        return self._velocity

    @property
    def velocity_range(self):
        """min, max of the absulute probe velocity during motion"""
        if self._velocity_range is None:
            minimum = np.min(self.velocity.iloc[self.start_idx:self.stop_idx].abs())
            self._velocity_range = SimpleNamespace(min=minimum, max=self.max_velocity)
        return self._velocity_range

    @property
    def avg_velocity(self):
        if self._avg_velocity is None:
            self._avg_velocity = abs(self.velocity.iloc[self.start_idx:self.stop_idx].mean())
        return self._avg_velocity

    @property
    def max_velocity(self):
        """Max velocity between start and stop"""
        if self._max_velocity is None:
            self._max_velocity = abs(self.velocity.iloc[self.start_idx:self.stop_idx].min())
        return self._max_velocity

    @property
    def distance_traveled(self):
        """Total distance traveled"""
        if self._distance_traveled is None:
            self._distance_traveled = abs(self.depth.max() - self.depth.min())
        return self._distance_traveled

    @property
    def avg_distance_traveled(self):
        """Average distance traveled"""
        if self._avg_distance_traveled is None:
            self._avg_distance_traveled = abs(self.depth.iloc[0:self.start_idx].mean() -
                                              self.depth.iloc[self.stop_idx:].mean())
        return self._avg_distance_traveled


    @property
    def distance_traveled_during_motion(self):
        """Total distance traveled between start and stop"""
        if self._distance_traveled_during_motion is None:
            self._distance_traveled_during_motion = abs(self.depth.iloc[self.start_idx] - self.depth.iloc[self.stop_idx])
        return self._distance_traveled_during_motion

    @property
    def has_upward_motion(self):
        """Contains upward motion in between the start and stop"""
        if self._has_upward_motion is None:
            self._has_upward_motion = False
            # crop the depth data and downsample for speedy check
            data = self.raw.iloc[self.start_idx:self.stop_idx]
            if len(data) > 1000:
                coarse = data.groupby(data.index // 200).first()
            else:
                coarse = data
            # loop and find any values greater than the current value
            for i, v in enumerate(coarse):
                upward = np.any(coarse.iloc[i:] > v + 5)
                if upward:
                    self._has_upward_motion = True
                    break

        return self._has_upward_motion


class BarometerDepth(DepthTimeseries):
    def __init__(self, *args, angle=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle = angle
    @property
    def depth(self):
        if self._depth is None:
            if self.stop_idx > self.start_idx:
                self._depth = get_constrained_baro_depth(self.raw, self.start_idx, self.stop_idx, method='nanmean')['baro']
                self._depth = self._depth.reindex(self.raw.index, method='nearest')
                # Adjust for an angle
                if self.angle is not None:
                    self._depth = self._depth / np.cos(np.pi * self.angle / 180)

            else:
                self._depth = pd.Series(index=self.raw.index, data=np.zeros_like(self.raw.values))
        return self._depth


class AccelerometerDepth(DepthTimeseries):
    @property
    def depth(self):
        if self._depth is None:
            valid = ~np.isnan(self.raw)
            self._depth = get_depth_from_acceleration(self.raw[valid])[self.raw.name]
            self._depth.iloc[self.stop_idx:] = self._depth.iloc[self.stop_idx]

        return self._depth
