import numpy as np
import pandas as pd
from scipy.signal import lfilter

def get_points_from_fraction(n_samples, fraction):
    """
    Return the nearest whole int from a fraction of the
    number of samples. Never returns 0.
    """
    idx = int(fraction * n_samples) or 1
    if idx == n_samples:
        idx = n_samples - 1
    return idx


def get_directional_mean(arr: np.array, fractional_basis: float = 0.01, direction='forward'):
    """
    Calculates the mean from a collection of points at the beginning or end of a dataframe
    """
    idx = get_points_from_fraction(len(arr), fractional_basis)
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
        series: pandas series of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start
        direction: forward to get a neutral bias at the start, backwards for the end.

    Returns:
        bias_adj: bias adjusted data to near zero
    """
    bias = get_directional_mean(series.values, fractional_basis=fractional_basis, direction=direction)
    bias_adj = series - bias
    return bias_adj


def get_normalized_at_border(series: pd.Series, fractional_basis: float = 0.01, direction='forward'):
    """
    Normalize a border by using the XX % of the data either at end of the data.
    e.g. the data was normalized by the mean of 1% of the beginning of the data.

    Args:
        series: pandas series of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start
        direction: Forward to norm the border at the start, backwards to norm at the end.
    Returns:
        border_norm: data by an average from one of the borders to nearly 1
    """
    border_avg = get_directional_mean(series, fractional_basis=fractional_basis, direction=direction)
    if border_avg != 0:
        border_norm = series / border_avg
    else:
        border_norm = series
    return border_norm

def merge_on_to_time(df_list, final_time):
    """
    Reindex the df fram the list onto a final time stamp
    """
    # Build dummy result in case no data is passed
    result = pd.DataFrame()

    # Merge everything else to it
    for i, df in enumerate(df_list):
        time_df = df.copy()
        if df.index.name != 'time':
            time_df = time_df.set_index('time')
        else:
            time_df = df.copy()

        data = time_df.reindex(time_df.index.union(final_time)).interpolate(method='cubic').reindex(final_time)
        if i == 0:
            result = data
        else:
            result = pd.merge_ordered(result, data, on='time', fill_method='cubic')

    # interpolate the nan's
    result = result.interpolate(method='nearest', limit_direction='both')
    return result


def merge_time_series(df_list):
    """
    Merges the other dataframes into the primary dataframe
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
            result = pd.merge_ordered(result, df, on='time', fill_method='cubic')

    # interpolate the nan's
    result = result.interpolate(method='index')
    return result


def remove_ambient(active, ambient, min_ambient_range=50, direction='forward'):
    """
    Attempts to remove the ambient signal from the active signal
    """
    amb_max = ambient.max()
    amb_min = ambient.min()
    if abs(amb_max - amb_min) > min_ambient_range:
        n = get_points_from_fraction(len(ambient), 0.01)
        amb = ambient.rolling(window=n, center=True, closed='both', min_periods=1).mean()
        norm_ambient = get_normalized_at_border(amb, direction=direction)
        norm_active = get_normalized_at_border(active, direction=direction)
        basis = get_directional_mean(active, direction=direction)
        clean = (norm_active - norm_ambient) * basis
        if clean.min() < 0 and 'backward' not in direction:
            clean = clean - clean.min()

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
    cols = [c for c in df.columns if c not in [dcol, 'time']]
    new = []
    # is the user request specific aggregation by column
    agg_col_specific = True if type(agg_method) == dict else False

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
        if agg_col_specific:
            for z,c in enumerate(cols):
                nr = getattr(df[c][ind], agg_method[c])()
                nr = pd.Series(data=nr, name=i, index=[c])
                if z == 0:
                    new_row = nr
                else:
                    new_row = pd.concat([new_row, nr])
        else:
            new_row = getattr(df[cols][ind], agg_method)(axis=0)
        new_row.name = i
        new_row[dcol] = d2
        new.append(new_row)
    result = pd.DataFrame.from_records(new)
    return result


def assume_no_upward_motion(series, method='nanmean', max_wind_frac=0.15):
    from .plotting import plot_ts

    i = 1
    result = series.copy()
    upfunc = getattr(np, method)
    while i < len(series):
        data = series.iloc[i]
        prev = series.iloc[i-1]

        # Check for upward movement
        if data > prev:
            # Find all the points
            ind = series.iloc[i-1:] >= prev

            rel_pos = np.where(ind)[0][-1]
            new_i = rel_pos + i-1
            new = upfunc(series.iloc[i-1:new_i])

            # grab last index, assign values
            result.iloc[i-1:new_i] = new
            ind = result.iloc[:new_i] <= new
            # Find only continuous areas where condition is true
            #continuous = (ind).astype(int).diff().abs().cumsum() == 0

            result.iloc[:new_i][ind] = new
            # ax = plot_ts(series, alpha=0.5, show=False, features=[i, new_i])
            # ax = plot_ts(result, ax=ax)
            # Watch out for mid values less than the new value
            #new_val_idx = np.where(ind)[0][0] + (i-1)
            #ind = result.iloc[:new_val_idx] < new
            #result.iloc[:new_val_idx][ind] = new
            #from .plotting import plot_ts

            #plot_ts(result, features=[i, new_i])

            i = new_i

        else:
            i += 1
    #from .plotting import plot_ts
    #result = result.rolling(window=max_n, center=True, closed='both', min_periods=1).mean()
    return result

def convert_force_to_pressure(force, tip_diameter_m, geom_adj=1):
    """
    Convert force data to pressure in KPa given the tip diameter and a tip shape adjustment
    for geometry differences.

    Args:
        force: Pandas Series in Newtons
        tip_diameter_m: Tip diameter in meters
        geom_adj: Adjustment factor to account for geometry diffs.
    Returns:
        pressure: instrument pressure series in kilopascals
    """
    area = np.pi * ((tip_diameter_m / 2) ** 2)
    # convert to pressure in Pascals
    pressure = force.div(area)
    # Adjust for shape and convert to kPa
    return pressure * geom_adj / 1000

def zfilter(series, fraction):
    """
    Zero phase filter
    """
    window = get_points_from_fraction(len(series), fraction)
    filter_coefficients = np.ones(window) / window

    # Apply the filter forward
    filtered_signal = lfilter(filter_coefficients, 1, series)

    # Apply the filter backward
    filtered_signal = lfilter(filter_coefficients, 1, filtered_signal[::-1])[::-1]

    return filtered_signal