import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np

from .decorators import time_series
from .adjustments import get_neutral_bias_at_border, assume_no_upward_motion
from .detect import get_acceleration_stop, get_acceleration_start, first_peak, nearest_valley, nearest_peak


@time_series
def get_depth_from_acceleration(acceleration_df: pd.DataFrame) -> pd.DataFrame:
    """
    Double integrate the acceleration to calculate a depth profile
    Assumes a starting position and velocity of zero.

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's without gravity

    Return:
        position_df: pandas Dataframe containing the same input axes plus magnitude of the result position
    """
    # Auto gather the x,y,z acceleration columns if they're there.
    acceleration_columns = [c for c in acceleration_df.columns if 'Axis' in c or 'acceleration' == c]

    # Convert from g's to m/s2
    g = -9.81
    acc = acceleration_df[acceleration_columns].mul(g)

    # Calculate position
    position_vec = {}
    for i, axis in enumerate(acceleration_columns):
        # Integrate acceleration to velocity
        v = cumtrapz(acc[axis].values, acc.index, initial=0)
        # Integrate velocity to position
        position_vec[axis] = cumtrapz(v, acc.index, initial=0)

    position_df = pd.DataFrame.from_dict(position_vec)
    position_df['time'] = acc.index
    position_df = position_df.set_index('time')

    # Calculate the magnitude if all the components are available
    if all([c in acceleration_columns for c in ['X-Axis', 'Y-Axis', 'Z-Axis']]):
        position_arr = np.array([position_vec['X-Axis'],
                                 position_vec['Y-Axis'],
                                 position_vec['Z-Axis']])
        position_df['magnitude'] = np.linalg.norm(position_arr, axis=0)
    return position_df


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
    baro_data = baro_depth.values
    top_search = baro_depth.iloc[:mid]
    default_top = np.where(top_search == top_search.max())[0][0]
    top = nearest_peak(baro_depth.values, start, default_index=default_top, height=-10, distance=100)
    # top = nearest_peak(df[baro].values, start, default_index=max_out, height=-0.1, distance=100)

    # Find valleys after, select closest to midpoint
    soft_stop = mid + int(0.1 * n_points)
    if soft_stop > len(baro_depth.index):
        soft_stop = len(baro_depth.index) - 1

    valley_search = baro_depth.iloc[mid:].values
    v_min = valley_search.min()
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
    # from .plotting import plot_constrained_baro
    # pos = get_depth_from_acceleration(df).mul(100)
    # pos = pos.reset_index()
    # plot_constrained_baro(df, result, const, pos, top, bottom, start, stop,
    #                       baro=baro, acc_axis=acc_axis)

    return constrained
