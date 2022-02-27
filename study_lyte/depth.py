import numpy as np
from scipy.integrate import cumtrapz


def get_depth_from_acceleration(acceleration, dt=None, series_time=None,
                                percent_basis=0.05, return_axis='Y-Axis'):
    """
    Use classic linear equations of motion to calculate position from
    acceleration
    Assuming the starting velocity is 0 and the starting position is also 0,
    then calculate the travel

    Args:
        acceleration: Pandas Dataframe containing X-Axis, Y-Axis, Z-Axis or acceleration
    """
    if series_time is None and 'time' == acceleration.index.name:
        series_time = acceleration.index
    elif series_time is None and 'time' in acceleration.columns:
        series_time = acceleration['time']

    # Auto gather the x,y,z acceleration columns if they're there.
    acceleration_columns = [c for c in acceleration.columns if 'Axis' in c]

    # If there are no Axis columns use acceleration
    if not acceleration_columns:
        # This was use for a long time just as a replacement for the Y-Axis
        acceleration_columns = ['acceleration']

    # Convert from g's to m/s2
    g = 9.81
    acc = acceleration[acceleration_columns] * g

    # Remove off local gravity
    idx = int(percent_basis * len(acc.index))
    print(idx)
    print(acc.iloc[0:idx].mean())

    acc = acc - acc.iloc[0:idx].mean()

    # Calculate position
    position_vec = {}
    for i, axis in enumerate(acceleration_columns):
        v = cumtrapz(acceleration[axis], series_time, initial=0)
        position_vec[axis] = cumtrapz(v, series_time, initial=0)


    # Calculate the magnitude from all the available components
    if return_axis == 'magnitude':
        position = (position_vec['X-Axis']**2 + position_vec['Y-Axis']**2 + position_vec['Z-Axis']**2)**0.5

    elif return_axis in acceleration.columns:
        position = position_vec[return_axis]
    else:
        choices = ', '.join(['X-Axis', 'Y-Axis', 'Z-Axis', 'magnitude'])
        raise ValueError(f'Invalid {return_axis} column requested, choices are {choices}')

    return position