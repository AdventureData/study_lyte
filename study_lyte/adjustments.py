import numpy as np
import pandas as pd


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
    bias = get_directional_mean(series, fractional_basis=fractional_basis, direction=direction)
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
            result = pd.merge_ordered(result, df, on='time')

    # interpolate the nan's
    result = result.interpolate(method='index')
    return result


def remove_ambient(active, ambient, min_ambient_range=100):
    """
    Attempts to remove the ambient signal from the active signal
    """
    amb_max = ambient.max()
    amb_min = ambient.min()
    if (amb_max - amb_min) > min_ambient_range:
        n = get_points_from_fraction(len(ambient), 0.01)
        amb = ambient.rolling(window=n, center=True, closed='both', min_periods=1).mean()
        norm_ambient = get_normalized_at_border(amb)
        norm_active = get_normalized_at_border(active)
        basis = get_directional_mean(active)
        clean = (norm_active - norm_ambient) * basis
        if clean.min() < 0:
            clean = clean - clean.min()

    else:
        clean = active
    # from .plotting import plot_ts
    # ax = plot_ts(clean, show=False)
    # ax = plot_ts(norm_active, ax=ax, alpha=0.5,  show=False)
    # ax = plot_ts(ambient, alpha=0.5,  ax=ax, show=True)

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
    new = []
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
        new.append(new_row)
    result = pd.DataFrame.from_records(new)
    return result


def assume_no_upward_motion(series, method='nanmean', max_wind_frac=0.15):
    i = 1
    result = series.copy()
    max_n = int(max_wind_frac*len(series))
    max_n = max_n or 1
    upfunc = getattr(np, method)
    while i < len(series):
        data = series.iloc[i]
        prev = series.iloc[i-1]

        # Check for upward movement
        if data > prev:
            # Find all the points
            ind = series.iloc[i-1:] >= prev
            rel_pos = np.where(ind)[0][0]
            new_i = rel_pos + i

            new = upfunc(series.iloc[i-1:new_i])
            # grap last index, assign values
            ind = result.iloc[i-1:] > new
            result.iloc[i-1:][ind] = new
            # Watch out for mid values less than the new value
            #new_val_idx = np.where(ind)[0][0] + (i-1)
            #ind = result.iloc[:new_val_idx] < new
            #result.iloc[:new_val_idx][ind] = new
            #from .plotting import plot_ts

            #plot_ts(result, features=[i, new_i])

            #i = new_i

        #else:
        i += 1
    #from .plotting import plot_ts
    result = result.rolling(window=max_n, center=True, closed='both', min_periods=1).mean()
    return result