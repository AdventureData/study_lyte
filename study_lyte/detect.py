import numpy as np
from scipy.signal import find_peaks
from .adjustments import get_neutral_bias_at_border, get_normalized_at_border
from .decorators import directional
from .plotting import plot_ts

@directional(check='search_direction')
def get_signal_event(signal_series, threshold=0.001, search_direction='forward', max_threshold=None, n_points=1):
    """
    Generic function for detecting relative changes in a given signal.

    Args:
        signal_series: Numpy Array or Pandas Series
        threshold: Float value of a min threshold of values to return as the event
        search_direction: string indicating which direction in the data to begin searching for event, options are forward/backward
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
    # if we have results, find the first match with npoints that meet the criteria
    if n_points > 1 and len(ind) > 0:
        npnts = n_points - 1
        id_diff = np.ones_like(ind) * 0
        id_diff[1:] = (ind[1:] - ind[0:-1])
        id_diff[0] = 1
        id_diff = np.abs(id_diff)
        spacing_ind = []
        # Determine if the last npoints are all 1 idx apart
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
    # if 'backward' in search_direction:
    #     event_idx = len(arr) - event_idx - 1

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
    pk_idx, pk_hgt = find_peaks(np.abs(accel_neutral), 0.3, distance=10)
    if pk_idx:
        max_ind = pk_idx[0]
    else:
        max_ind = 1

    acceleration_start = get_signal_event(accel_neutral[0:max_ind+1], threshold=threshold, max_threshold=max_threshold,
                                          n_points=int(0.005 * len(acceleration)),
                                          search_direction='forward')
    ax = plot_ts(accel_neutral, events=[('start', acceleration_start)], features=pk_idx)
    return acceleration_start


def get_acceleration_stop(acceleration, fractional_basis=0.02, threshold=-0.03, max_threshold=0.01):
    """
    Returns the index of the last value that has a relative change greater than the
    threshold of absolute normalized signal
    Args:
        acceleration:pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold: Float in g's for
        max_threshold: Max value between the max of the signal and the end to be considered for stop criteria

    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    acceleration = acceleration[~np.isnan(acceleration)]

    # remove gravity
    accel_neutral = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis, direction='backward')
    peaks = find_peaks(np.abs(accel_neutral), height=0.3, distance=5)
    ind = peaks[0][-1]

    # Isolate the area of the signal known to have the stop
    sig = accel_neutral[ind:]

    # Use the number of points variably
    n_points = int(0.01 * len(sig))
    if n_points > 200:
        n_points = 200

    event = get_signal_event(sig, threshold=threshold, max_threshold=max_threshold, n_points=n_points,
                             search_direction='backward')

    if event == 0:
        acceleration_stop = len(acceleration) - 1
    else:
        acceleration_stop = ind + event

    return acceleration_stop


def get_nir_surface(ambient, active, fractional_basis=0.01, threshold=0.1):
    """
    Using the active and ambient NIR, estimate the index at when the probe was in the snow.
    The ambient signal is expected to receive less and less light as it enters into the snowpack,
    whereas the active should receive more. This function calculates the first value of the
    difference of the two signals should be the snow surface.

    Args:
        ambient: Numpy Array or pandas Series of the ambient NIR signal
        active: Numpy Array or pandas Series of the active NIR signal
        fractional_basis: Float of begining data to use as the normalization
        threshold: Float threshold value for meeting the snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
    ambient = ambient.values
    active = active.values

    amb_norm = get_normalized_at_border(ambient, fractional_basis=fractional_basis)
    act_norm = get_normalized_at_border(active, fractional_basis=fractional_basis)
    diff = abs(act_norm - amb_norm)
    surface = get_signal_event(diff, threshold=threshold, search_direction='forward')
    return surface


def get_nir_stop(active, n_points_for_basis=1000, threshold=0.01):
    """
    Often the NIR signal shows the stopping point of the probe by repeated data.
    This looks at the active signal to estimate the stopping point
    """
    bias = active[-1 * n_points_for_basis:].min()
    norm = active - bias
    norm = abs(norm / norm.max())
    stop = get_signal_event(norm, threshold=threshold, search_direction='backward')

    return stop
