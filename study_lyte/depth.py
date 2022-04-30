import pandas as pd
from scipy.integrate import cumtrapz
from scipy import signal
import numpy as np
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from .decorators import time_series
from .adjustments import get_neutral_bias_at_border


@time_series
def get_depth_from_acceleration(acceleration_df: pd.DataFrame, fractional_basis: float = 0.01) -> pd.DataFrame:
    """
    Double integrate the acceleration to calcule a depth profile
    Assumes a starting position and velocity of zero.

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's
        fractional_basis: fraction of the begining of data to calculate a bias adjustment

    Returns:

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis or acceleration
    Return:
        position_df: pandas Dataframe containing the same input axes plus magnitude of the result position
    """
    # Auto gather the x,y,z acceleration columns if they're there.
    acceleration_columns = [c for c in acceleration_df.columns if 'Axis' in c or 'acceleration' == c]

    # Convert from g's to m/s2
    g = -9.81
    acc = acceleration_df[acceleration_columns].mul(g)

    # Remove off local gravity
    acc = get_neutral_bias_at_border(acc, fractional_basis)

    # Calculate position
    position_vec = {}
    for i, axis in enumerate(acceleration_columns):
        # Integrate acceleration to velocity
        v = cumtrapz(acc[axis].values, acc.index, initial=0)
        # Integrate velocity to postion
        position_vec[axis] = cumtrapz(v, acc.index, initial=0)

    position_df = pd.DataFrame.from_dict(position_vec)
    position_df['time'] = acc.index
    position_df.set_index('time', inplace=True)

    # Calculate the magnitude if all the components are available
    if all([c in acceleration_columns for c in ['X-Axis', 'Y-Axis', 'Z-Axis']]):
        position_arr = np.array([position_vec['X-Axis'],
                                position_vec['Y-Axis'],
                                position_vec['Z-Axis']])
        position_df['magnitude'] = np.linalg.norm(position_arr, axis=0)
    return position_df


@time_series
def get_average_depth(df, acc_axis='Y-Axis', depth_column='depth') -> pd.DataFrame:
    """
    Calculates the average between the barometer and the accelerometer profile
    """
    depth = get_depth_from_acceleration(df[[acc_axis]]) * 100
    depth[depth_column] = get_neutral_bias_at_border(df[depth_column], fractional_basis=0.005)
    depth['depth'] = depth[[depth_column, acc_axis]].mean(axis=1)
    depth['depth'] = get_neutral_bias_at_border(depth['depth'], fractional_basis=0.005)
    return depth[['depth']]


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
def get_depth_from_acceleration_w_infill(df: pd.DataFrame, max_g=3, percent_basis: float = 0.05) -> pd.DataFrame:
    """
    Calculating depth from acceleration is great without any crusts. Here
    the barometer is used to infill the maxed out accelerometer data to
    provide a more reliable profile.

    Args:
        df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's and depth from barom in cm
        max_g: maximum acceleration in gs to filter out and replace with barometer data

    Returns:

    """
    baro_df = df[['depth']].copy(deep=True)
    baro_df['depth'] = baro_df['depth'].div(100)
    baro_df = get_neutral_bias_at_border(baro_df, fractional_basis=0.005)

    # Copy out the accelerometer, convert to g's, save where max out happens
    g = -9.81
    acc_df = df[['Y-Axis']].copy(deep=True).mul(g)
    infill_idx = df['Y-Axis'].abs() > abs(g * max_g)
    acc_df = get_neutral_bias_at_border(acc_df, fractional_basis=0.01)

    # Calculate the velocity of barometer, clean it up
    baro_df['velocity'] = np.gradient(baro_df['depth'], baro_df.index)
    baro_df['velocity'] = signal.medfilt(baro_df['velocity'], 501)
    baro_df['velocity'] = baro_df['velocity'].rolling(window=600).mean()

    # Get the velocity from acceleration
    acc_df['velocity'] = cumtrapz(acc_df['Y-Axis'].values,
                                  acc_df.index, initial=0)

    # Infill the velocity
    infill_df = acc_df.copy(deep=True)
    infill_df.loc[infill_idx, 'velocity'] = baro_df.loc[infill_idx, 'velocity']

    # Calculate the position from the realigned warped average profile
    infill_df['depth'] = cumtrapz(infill_df['velocity'], x=infill_df['velocity'].index, initial=0)
    return infill_df


# IN PROGRESS
@time_series
def get_hybrid_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculating depth from acceleration is great without any crusts. Here
    the barometer is used to infill the maxed out accelerometer data to
    provide a more reliable profile.

    Args:
        accelerometer_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's and depth from barom in cm

    Returns:

    """
    baro_df = df[['depth']].copy(deep=True)
    acc_df = df[['Y-Axis']].copy(deep=True).mul(9.81)

    # Calculate the velocity of barometer, clean it up
    baro_df['velocity'] = np.gradient(baro_df['depth'], baro_df.index)
    baro_df['velocity'] = signal.medfilt(baro_df['velocity'], 501)
    baro_df['velocity'] = baro_df['velocity'].rolling(window=600).mean()

    # Get the velocity from acceleration
    acc_df['velocity'] = cumtrapz(acc_df['Y-Axis'].values,
                                  acc_df.index, initial=0)

    # Match the velocity profiles
    bv = baro_df['velocity'].dropna().values
    av = acc_df['velocity'].dropna().values
    distance, path = fastdtw(av, bv, dist=euclidean)

    # Grab the stretched data
    stretched_acc = [av[idx[0]] for idx in path]
    stretched_baro = [bv[idx[1]] for idx in path]
    stretched_time = [df.index[idx[0]] for idx in path]

    # Calculate a a mean velocity curve considering the best alignment of the two datasets
    mean_velocity = np.array([stretched_acc, stretched_baro]).mean(axis=0)

    # Calculate an average depth and group on time which has duplicates currently
    avg_velocity = pd.DataFrame.from_dict({'time': stretched_time, 'Y-Axis': mean_velocity})
    avg_velocity = avg_velocity.groupby(by='time').mean()

    # Calculate the position from the realigned warped average profile
    warped_mean_depth = cumtrapz(avg_velocity['Y-Axis'], x=avg_velocity.index, initial=0)
    return warped_mean_depth

