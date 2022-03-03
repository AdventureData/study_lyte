import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np

def get_depth_from_acceleration(acceleration_df: pd.DataFrame, percent_basis: float = 0.05) -> pd.DataFrame:
    """
    Double integrate the acceleration to calcule a depth profile
    Assumes a starting position and velocity of zero.

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis in g's
        percent_basis: fraction of the begining of data to calculate a bias adjustment

    Returns:

    Args:
        acceleration_df: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis or acceleration
    Return:
        position_df: pandas Dataframe containing the same input axes plus magnitude of the result position
    """
    if acceleration_df.index.name != 'time' and 'time' not in acceleration_df.columns:
        raise ValueError(f"Acceleration data requires a 'time' column or index to calculate depth!")

    if 'time' in acceleration_df.columns:
        acceleration_df.set_index('time', inplace=True)

    # Auto gather the x,y,z acceleration columns if they're there.
    acceleration_columns = [c for c in acceleration_df.columns if 'Axis' in c or 'acceleration' == c]

    # Convert from g's to m/s2
    g = 9.81
    acc = acceleration_df[acceleration_columns] * g

    # Remove off local gravity
    idx = int(percent_basis * len(acc.index))
    acc = acc - acc.iloc[0:idx].mean()

    # Calculate position
    position_vec = {}
    for i, axis in enumerate(acceleration_columns):
        # Integrate acceleration to velocity
        v = cumtrapz(acc[axis].values, acc.index, initial=0)
        # Integrate velocity to postion
        position_vec[axis] = cumtrapz(v, acc.index, initial=0)

    position_df = pd.DataFrame.from_dict(position_vec)

    # Calculate the magnitude if all the components are available
    if all([c in ['X-Axis', 'Y-Axis', 'Z-Axis'] for c in acceleration_columns]):
        position_arr = np.array([position_vec['X-Axis'],
                                position_vec['Y-Axis'],
                                position_vec['Z-Axis']])
        position_df['magnitude'] = np.linalg.norm(position_arr, axis=0)

    return position_df
