import numpy as np
from scipy.signal import find_peaks, argrelextrema

from .adjustments import get_neutral_bias_at_border, get_normalized_at_border
from .decorators import directional
from .plotting import  plot_ts

def first_peak(arr, default_index=1, **find_peak_kwargs):
    """
    Finds peaks and a return the first found. if none are found
    return the default index
    """
    pk_idx, pk_hgt = find_peaks(np.abs(arr), **find_peak_kwargs)
    if len(pk_idx) > 0:
        pk = pk_idx[0]
    else:
        pk = default_index
    return pk


def nearest_peak(arr, nearest_to_index, default_index=0, **find_peak_kwargs):
    """Find the nearest peak to a designated point"""
    pk_idx, pk_hgt = find_peaks((arr + arr.min()), **find_peak_kwargs)
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

    # Invert the index
    if 'backward' in search_direction:
        ind = len(arr) - ind - 1
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
        event_idx = ind[-1]

    return event_idx


def get_acceleration_start(acceleration, fractional_basis: float = 0.01, threshold=-0.01, max_threshold=0.02):
    """
    Returns the index of the first value that has a relative change
    Args:
        acceleration: np.array or pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold: relative minimum change to indicate start
        max_threshold: Maximum allowed threshold to be considered a start
    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    acceleration = acceleration[~np.isnan(acceleration)]

    # Get the neutral signal between start and the max
    accel_neutral = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis, direction='forward')
    max_ind = first_peak(np.abs(accel_neutral), height=0.3, distance=10)

    acceleration_start = get_signal_event(accel_neutral[0:max_ind+1], threshold=threshold, max_threshold=max_threshold,
                                          n_points=int(0.005 * len(acceleration)),
                                          search_direction='forward')
    return acceleration_start


def get_acceleration_stop(acceleration, fractional_basis=0.02, height=0.3, distance=10):
    """
    Returns the index of the last value that has a relative change greater than the
    threshold of absolute normalized signal
    Args:
        acceleration:pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        height: Float in g's for minimum peak findable
        distance: Minimum distance between peaks
    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    acceleration = acceleration[~np.isnan(acceleration)]

    # remove gravity
    accel_neutral = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis,
                                               direction='forward')
    # Find the first real negative peak starting at the end
    ind = first_peak(-1 * accel_neutral[::-1], default_index=0, height=height, distance=distance)
    if ind == 0:
        acceleration_stop = len(accel_neutral) - 1
    else:
        acceleration_stop = len(accel_neutral) - ind + 1

    return acceleration_stop


def get_nir_surface(ambient, active, fractional_basis=0.01, threshold=0.05):
    """
    Using the active and ambient NIR, estimate the index at when the probe was in the snow.
    The ambient signal is expected to receive less and less light as it enters into the snowpack,
    whereas the active should receive more. This function calculates the first value of the
    difference of the two signals should be the snow surface.

    Args:
        ambient: Numpy Array or pandas Series of the ambient NIR signal
        active: Numpy Array or pandas Series of the active NIR signal
        fractional_basis: Float of beginning data to use as the normalization
        threshold: Float threshold value for meeting the snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
    ambient = ambient.rolling(20).mean().values
    active = active.rolling(20).mean().values

    amb_norm = get_normalized_at_border(ambient, fractional_basis=fractional_basis)
    act_norm = get_normalized_at_border(active, fractional_basis=fractional_basis)
    diff = amb_norm - act_norm
    gradient = np.sign(diff)
    surface = get_signal_event(diff, threshold=threshold, search_direction='backward')
    ax = plot_ts(diff, events=[('surface', surface)], show=True)
    ax = plot_ts(gradient, ax=ax, show=True)
    return surface


def get_nir_stop(active, n_points_for_basis=1000, threshold=0.01):
    """
    Often the NIR signal shows the stopping point of the probe by repeated data.
    This looks at the active signal to estimate the stopping point
    """
    bias = active[-1 * n_points_for_basis:].min()
    norm = active - bias
    norm = abs(norm / norm.max())
    stop = get_signal_event(norm, threshold=threshold, search_direction='forward')

    return stop
