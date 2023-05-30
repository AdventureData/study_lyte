import numpy as np
from scipy.signal import find_peaks, argrelextrema

from .adjustments import get_neutral_bias_at_border, get_normalized_at_border, get_points_from_fraction
from .decorators import directional


def first_peak(arr, default_index=1, **find_peak_kwargs):
    """
    Finds peaks and a return the first found. if none are found
    return the default index
    """
    pk_idx, pk_hgt = find_peaks(arr, **find_peak_kwargs)
    if len(pk_idx) > 0:
        pk = pk_idx[0]
    else:
        pk = default_index
    return pk


def nearest_peak(arr, nearest_to_index, default_index=0, **find_peak_kwargs):
    """Find the nearest peak to a designated point"""
    pk_idx, pk_hgt = find_peaks(arr, **find_peak_kwargs)
    if len(pk_idx) > 0:
        nearest_val = pk_idx[(np.abs(pk_idx - nearest_to_index)).argmin()]
    else:
        nearest_val = default_index
    return nearest_val


def nearest_valley(arr, nearest_to_index, default_index=1):
    """Find the nearest valley closest to a designated point"""
    valleys = argrelextrema(arr, np.less)[0]
    if len(valleys) > 0:
        nearest_val = valleys[(np.abs(valleys - nearest_to_index)).argmin()]
    else:
        nearest_val = default_index
    return nearest_val


@directional(check='search_direction')
def get_signal_event(signal_series, threshold=0.001, search_direction='forward', max_threshold=None, n_points=1):
    """
    Generic function for detecting relative changes in a given signal.

    Args:
        signal_series: Numpy Array or Pandas Series
        threshold: Float value of a min threshold of values to return as the event
        search_direction: string indicating which direction in the data to begin searching for event, options are
                        forward/backward
        max_threshold: Float value of a max threshold that events have to be under to be an event
        n_points: Number of points in a row meeting threshold criteria to be an event.

    Returns:
        event_idx: Integer of the index where values meet the threshold criteria
    """
    # n points can't be 0
    n_points = n_points or 1
    # Parse whether to work with a pandas Series
    if hasattr(signal_series, 'values'):
        sig = signal_series.values
    # Assume Numpy array
    else:
        sig = signal_series
    arr = sig

    # Invert array if backwards looking
    if 'backward' in search_direction:
        arr = sig[::-1]

    # Find all values between threshold and max threshold
    idx = arr >= threshold
    if max_threshold is not None:
        idx = idx & (arr < max_threshold)
    # Parse the indices
    ind = np.argwhere(idx)
    ind = np.array([i[0] for i in ind])

    # if we have results, find the first match with n points that meet the criteria
    if n_points > 1 and len(ind) > 0:
        npnts = n_points - 1
        id_diff = np.ones_like(ind) * 0
        id_diff[1:] = (ind[1:] - ind[0:-1])
        id_diff[0] = 1
        id_diff = np.abs(id_diff)
        spacing_ind = []

        # Determine if the last n points are all 1 idx apart
        for i, ix in enumerate(ind):
            if i >= npnts:
                test_arr = id_diff[i - npnts:i + 1]
                if all(test_arr == 1):
                    spacing_ind.append(ix)
        ind = spacing_ind

    # If no results are found, return the first index the series
    if len(ind) == 0:
        event_idx = 0
    else:
        # Return the first value matching the conditions
        event_idx = ind[-1]

    # Invert the index
    if 'backward' in search_direction:
        event_idx = len(arr) - 1 - event_idx

    return event_idx


def get_acceleration_start(acceleration, threshold=-0.01, max_threshold=0.02):
    """
    Returns the index of the first value that has a relative change
    Args:
        acceleration: np.array or pandas series of acceleration without gravity
        threshold: relative minimum change to indicate start
        max_threshold: Maximum allowed threshold to be considered a start
    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    accel_neutral = acceleration[~np.isnan(acceleration)]

    # Get the neutral signal between start and the max
    max_ind = first_peak(np.abs(accel_neutral), height=0.3, distance=10)
    n_points = get_points_from_fraction(len(acceleration), 0.005)
    acceleration_start = get_signal_event(accel_neutral[0:max_ind+1], threshold=threshold, max_threshold=max_threshold,
                                          n_points=n_points,
                                          search_direction='forward')
    return acceleration_start


def get_acceleration_stop(acceleration, threshold=0.1, max_threshold=0.3):
    """
    Returns the index of the last value that has a relative change greater than the
    threshold of absolute normalized signal
    Args:
        acceleration:pandas series of acceleration data
        threshold: Float in g's for minimum to consider
        max_threshold: Max threshold to consider
    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    accel_neutral = -1 * acceleration[~np.isnan(acceleration)]
    n_points = get_points_from_fraction(len(acceleration), 0.005)
    max_ind = first_peak(accel_neutral[::-1], height=0.3, distance=10)
    max_ind = len(acceleration) - max_ind - 1

    acceleration_stop = get_signal_event(accel_neutral[max_ind:], threshold=threshold,
                                         max_threshold=max_threshold,
                                         n_points=None,
                                         search_direction='backward')
    acceleration_stop = acceleration_stop + max_ind
    return acceleration_stop


def get_nir_surface(clean_active, threshold=0.05, max_threshold=0.1):
    """
    Using the cleaned active, estimate the index at when the probe was in the snow.

    Args:
        clean_active: Numpy Array or pandas Series of the clean NIR signal
        threshold: Float minimum relative percent change threshold value for a snow surface event
        max_threshold: Float maximum relative percent change threshold value for a snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
    clean_norm = clean_active / clean_active.max()
    clean_norm = clean_norm - abs(clean_norm).min()
    surface = get_signal_event(clean_norm, search_direction='forward', threshold=threshold,
                               max_threshold=max_threshold)
    return surface


def get_nir_stop(active, fractional_basis=0.05, max_threshold=0.01, threshold=-0.01):
    """
    Often the NIR signal shows the stopping point of the probe by repeated data.
    This looks at the active signal to estimate the stopping point
    """
    # Perform a removal of ambient but using the end of the data.
    n = get_points_from_fraction(len(active), 0.05)
    border_fract = 0.3
    norm_active = get_normalized_at_border(active, fractional_basis=border_fract, direction='backward')
    norm_active = norm_active.rolling(window=n, center=True, closed='both', min_periods=1).mean()
    norm_active = norm_active - 1

    ind = np.where(norm_active == norm_active.max())[0][0]
    data = norm_active.iloc[ind:]
    # diff = diff.rolling(window=n, center=True, closed='both', min_periods=1).median()

    n_points = get_points_from_fraction(len(data), fractional_basis)
    stop = get_signal_event(data, search_direction='backward', threshold=threshold,
                            max_threshold=max_threshold, n_points=n_points)
    stop += ind

    # from .plotting import plot_ts, plt
    # ax = plot_ts(norm_active, events=[('stop', stop), ('avg_tail', len(norm_active) - n)],
    #              color='red', show=False)
    # ax.axhline(threshold)
    # ax.axhline(max_threshold)
    # plt.show()

    return stop

