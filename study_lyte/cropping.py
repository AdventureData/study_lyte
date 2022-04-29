from .detect import get_acceleration_start, get_acceleration_stop, get_nir_surface
from .decorators import time_series
import pandas as pd


@time_series
def crop_to_motion(df: pd.DataFrame, detect_col='Y-Axis', start_kwargs, stop_kwargs) -> pd.DataFrame:
    """
    Crop the dataset to the only the motion as seen by the
    accelerometer

    Args:
        df: pd.DataFrame containing the column specified for acceleration
        detect_col: Column name to use to determine the start/stop of motion
        start_kwargs: Dict of keyword arguments to pass on to detect.get_acceleration_start
        stop_kwargs: Dict of  keyword arguments to pass on to detect.get_acceleration_stop

    Returns:
        cropped: pd.Dataframe cropped to the time period where motion start/stopped
    """
    start = get_acceleration_start(df[detect_col], **start_kwargs)
    stop = get_acceleration_stop(df[detect_col], **stop_kwargs)

    cropped = df.loc[start:stop, df.columns]
    return cropped


@time_series
def crop_to_snow(df: pd.DataFrame, active_col='Sensor2', ambient_col='Sensor3', **kwargs) -> pd.DataFrame:
    """
    Crop the dataset to the only the motion as seen by the
    accelerometer

    Args:
        df: pd.DataFrame containing the column specified for acceleration
        active_col: Column name containing active nir data
        ambient_col: Column name containing ambient nir data
        kwargs: Other keyword arguments to pass on to detect.get_nir_surface

    Returns:
        cropped: pd.Dataframe cropped to the time period where motion start/stopped
    """
    surface = get_nir_surface(df)
    cropped = df.loc[surface:, df.columns]
    return cropped

