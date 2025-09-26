import numpy as np
from scipy.signal import argrelextrema

from .adjustments import (get_neutral_bias_at_border, get_normalized_at_border, get_points_from_fraction, get_neutral_bias_at_index,zfilter)
from .decorators import directional

def find_nearest_value_index(search_value, series):
    """
    Given an array and a value, this function finds the index of the
    value closest to the search value
    """
    idx = np.abs(search_value - series).argmin()
    return idx


def find_peaks(arr, height=None, distance=1):
    """
    Basic replacement for scipy.signal.find_peaks.
    Finds indices where arr[i] > arr[i-1] and arr[i] > arr[i+1].
    Supports optional height and minimum distance between peaks.
    """
    peaks = []
    arr = np.asarray(arr)
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if height is not None and arr[i] < height:
                continue
            if peaks and (i - peaks[-1]) < distance:
                continue
            peaks.append(i)
    return np.array(peaks), arr[peaks]


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


def find_valleys(arr):
    """
    Finds indices where arr[i] < arr[i-1] and arr[i] < arr[i+1].
    """
    arr = np.asarray(arr)
    valleys = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            valleys.append(i)
    return np.array(valleys)


def nearest_valley(arr, nearest_to_index, default_index=1):
    """Find the nearest valley closest to a designated point"""
    valleys = find_valleys(arr)
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
        event_idx = None
    else:
        # Return the first value matching the conditions
        event_idx = ind[-1]

    # Invert the index
    if 'backward' in search_direction and event_idx is not None:
        event_idx = len(arr) - 1 - event_idx
        # if event_idx == 0:
        #     event_idx = None


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
    if acceleration_start is None:
        acceleration_start = 0
    return acceleration_start


def get_acceleration_stop(acceleration, threshold=-0.2, max_threshold=0.1):
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

    min_idx = np.argwhere(acceleration == acceleration.min())[0][0]
    max_idx = np.argwhere(acceleration == acceleration.max())[0][0]
    # Large impact early in during the accelerating down
    if min_idx < max_idx:
        # Find the deceleration later than the maximum deceleration
        search_start = np.argwhere(acceleration[max_idx:] == acceleration[max_idx:].min())[0][0]
        search_start += max_idx
    else:
        # Use the farthest deceleration
        search_start = min_idx

    n = get_points_from_fraction(len(acceleration[search_start:]), 0.05, maximum=1000)
    acceleration_stop = get_signal_event(acceleration[search_start:], threshold=threshold,
                                         max_threshold=max_threshold,
                                         n_points=n,
                                         search_direction='backward')

    if acceleration_stop is None or acceleration_stop == 0:
        acceleration_stop = len(acceleration) - 1
    else:
        acceleration_stop = acceleration_stop + search_start
    return acceleration_stop


def get_nir_surface(clean_active, threshold=30, max_threshold=None):
    """
    Using the cleaned active, estimate the index at when the probe was in the snow.

    Args:
        clean_active: Numpy Array or pandas Series of the clean NIR signal
        threshold: Float minimum relative percent change threshold value for a snow surface event
        max_threshold: Float maximum relative percent change threshold value for a snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
    # n = get_points_from_fraction(len(clean_active), 0.01)
    # Normalize by data unaffected by ambient
    neutral = get_neutral_bias_at_border(clean_active)

    # Retrieve a likely candidate under challenging ambient conditions
    window = get_points_from_fraction(len(neutral), 0.01)
    diff = neutral.rolling(window=window).std().values

    # Detect likely candidate normal ambient conditions
    surface = get_signal_event(diff, search_direction='backward', threshold=threshold,
                               max_threshold=max_threshold, n_points=1)
    # No surface found and all values met criteria
    if surface == len(neutral)-1 or surface is None:
        surface = 0
    # from .plotting import plot_nir_surface
    # plot_nir_surface(neutral, diff, surface)
    return surface


def get_nir_stop(active, fractional_basis=0.05, max_threshold=0.008, threshold=-0.05):
    """
    Often the NIR signal shows the stopping point of the probe by repeated data.
    This looks at the active signal to estimate the stopping point
    """
    n = get_points_from_fraction(len(active), 0.1)
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
    if stop is not None:
        stop += ind

    return stop


def get_sensor_start(signal, fractional_basis=0.05, max_threshold=0.05, threshold=-0.05):
    """
    Before entering the snow we don't see much dynamic signal. This detects the first change in the signal
    """
    ind = np.where(signal == signal.min())[0][0]
    n_points = get_points_from_fraction(len(signal), 0.01)
    data = signal[:ind]
    norm_signal = get_normalized_at_border(data, fractional_basis=fractional_basis, direction='forward') - 1
    first_change = get_signal_event(norm_signal, search_direction='forward', threshold=threshold,
                            max_threshold=max_threshold, n_points=n_points)
    return first_change


def get_ground_strike(signal, stop_idx):
    """
    The probe hits ground somtimes before we detect stop.
    """
    buffer = get_points_from_fraction(len(signal), 0.12)
    start = stop_idx - buffer
    start = start if start > 0 else 0
    end = stop_idx + buffer
    end = end if end < len(signal) else len(signal)-1
    rel_stop = stop_idx - start

    sig_arr = signal[start:end]
    window = get_points_from_fraction(len(sig_arr), 0.01)
    diff = sig_arr.rolling(window=window).std().values
    diff = get_neutral_bias_at_border(diff, direction='backward')

    # Large change in signal
    impact = get_signal_event(diff, threshold=150, max_threshold=1000, n_points=1, search_direction='forward')

    # Large chunk of data that's the same near the stop
    norm1 = get_neutral_bias_at_index(sig_arr, rel_stop+buffer).values
    n_points = get_points_from_fraction(len(norm1), 0.1)
    long_press = get_signal_event(norm1, threshold=-10000, max_threshold=150, n_points=n_points, search_direction='backward')
    tol = get_points_from_fraction(len(norm1), 0.1)

    ground = None
    if impact is not None:
        impact += start
    if long_press is not None:
        long_press += start

    if long_press is not None and impact is not None:
        if (long_press-tol) <= impact <= (long_press+tol):
            ground = impact

    # from .plotting import  plot_ground_strike, plot_ts
    # plot_ground_strike(signal, diff, norm1, start, stop_idx, impact, long_press,ground)

    return ground