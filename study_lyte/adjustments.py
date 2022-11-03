import numpy as np
import pandas as pd


def get_directional_mean(arr: np.array, fractional_basis: float = 0.01, direction='forward'):
    """
    Calculates the mean from a collection of points at the beginning or end of a dataframe
    """
    idx = int(fractional_basis * len(arr)) or 1
    if direction == 'forward':
        avg = arr[0:idx].mean()
    elif direction == 'backward':
        avg = arr[-1*idx:].mean()
    else:
        raise ValueError('Invalid Direction used, Use either forward or backward.')
    return avg


def get_neutral_bias_at_border(df: pd.DataFrame, fractional_basis: float = 0.01, direction='forward'):
    """
    Bias adjust the series data by using the XX % of the data either at the front of the data
    or the end of the .
    e.g. 1% of the data is averaged and subtracted.

    Args:
        df: pandas dataframe of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start

    Returns:
        bias_adj: bias adjusted data to near zero
    """
    bias = get_directional_mean(df, fractional_basis=fractional_basis, direction=direction)
    bias_adj = df - bias
    return bias_adj


def get_normalized_at_border(df: pd.DataFrame, fractional_basis: float = 0.01, direction='forward'):
    """
    Normalize a border by using the XX % of the data either at end of the data.
    e.g. the data was normalized by the mean of 1% of the beginning of the data.

    Args:
        df: pandas dataframe of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start

    Returns:
        border_norm: data by an average from one of the borders to nearly 1
    """
    border_avg = get_directional_mean(df, fractional_basis=fractional_basis, direction=direction)
    border_norm = df / border_avg
    return border_norm


def merge_time_series(df_list):
    """
    Merges the other dataframes into the primary datafrane
    which set the resolution for the other dataframes. The final
    result is interpolated to eliminate nans.

    Args:
        df_list: List of pd Dataframes to be merged and interpolated

    Returns:
        result: pd.DataFrame containing the interpolated results all merged
                into the same dataframe using the high resolution
    """
    # Build dummy result in case no data is passed
    result = pd.DataFrame()

    # Merge everything else to it
    for i, df in enumerate(df_list):
        if i == 0:
            result = df.copy()
        else:
            result = pd.merge_ordered(result, df, on='time')

    # interpolate the nan's
    result = result.interpolate(method='index')
    return result
