import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np
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


