import numpy as np
import pandas as pd


def get_directional_mean(arr: np.array, fractional_basis: float = 0.01, direction='forward'):
    """
    Calculates the mean from a collection of points at the beginning or end of a dataframe
    """
    idx = int(fractional_basis * len(arr)) or 1
    if direction == 'forward':
        avg = np.nanmean(arr[:idx])
    elif direction == 'backward':
        avg = np.nanmean(arr[-1*idx:])
    else:
        raise ValueError('Invalid Direction used, Use either forward or backward.')
    return avg


def get_neutral_bias_at_border(series: pd.Series, fractional_basis: float = 0.01, direction='forward'):
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
    bias = get_directional_mean(series, fractional_basis=fractional_basis, direction=direction)
    bias_adj = series - bias
    return bias_adj


def get_normalized_at_border(series: pd.Series, fractional_basis: float = 0.01, direction='forward'):
    """
    Normalize a border by using the XX % of the data either at end of the data.
    e.g. the data was normalized by the mean of 1% of the beginning of the data.

    Args:
        df: pandas series of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start

    Returns:
        border_norm: data by an average from one of the borders to nearly 1
    """
    border_avg = get_directional_mean(series, fractional_basis=fractional_basis, direction=direction)
    border_norm = series / border_avg
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


def remove_ambient(active, ambient, min_ambient_range=100):
    """
    Attempts to remove the ambient signal from the active signal
    """
    if (ambient.max() - ambient.min()) > min_ambient_range:
        norm_ambient = get_normalized_at_border(ambient)
        norm_active = get_normalized_at_border(active)
        basis = get_directional_mean(active)
        clean = (norm_active - norm_ambient) * basis
    else:
        clean = active
    return clean


def apply_calibration(series, coefficients, minimum=None, maximum=None):
    """
    Apply any calibration using poly1d
    """
    poly = np.poly1d(coefficients)
    result = poly(series)

    if maximum is not None:
        result[result > maximum] = maximum
    if minimum is not None:
        result[result < minimum] = minimum

    return result


def aggregate_by_depth(df, new_depth, df_depth_col='depth', agg_method='mean'):
    """
    Aggregate the dataframe by the new depth using whatever method
    provided. Data in the new depth is considered to be the bottom of
    the aggregation e.g. 10, 20 == 0-10, 11-20 etc
    Depth data must be monotonic.
    new_depth data much be coarser than current depth data
    """
    if new_depth[-1] < 0:
        surface_datum = True
    else:
        surface_datum = False
    if df.index.name is not None:
        df = df.reset_index()
    dcol = df_depth_col
    result = pd.DataFrame(columns=df.columns)
    cols = [c for c in df.columns if c != dcol]
    for i, d2 in enumerate(new_depth):
        # Find previous depth value for comparison
        if i == 0:
            d1 = df[dcol].iloc[0]
        else:
            d1 = new_depth[i-1]

        # Manage negative depths
        if surface_datum:
            ind = df[dcol] >= d2
            if i == 0:
                ind = ind & (df[dcol] <= d1)
            else:
                ind = ind & (df[dcol] < d1)
        else:
            ind = df[dcol] <= d2
            if i == 0:
                ind = ind & (df[dcol] >= d1)
            else:
                ind = ind & (df[dcol] > d1)
        new_row = getattr(df[cols][ind], agg_method)(axis=0)
        new_row.name = i
        new_row[dcol] = d2
        result = result.append(new_row)

    return result
