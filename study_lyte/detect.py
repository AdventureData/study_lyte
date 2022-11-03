import numpy as np

from .adjustments import get_neutral_bias_at_border, get_normalized_at_border
from .decorators import directional


@directional(check='search_direction')
def get_signal_event(signal_series, threshold=0.001, search_direction='forward', max_theshold=None, n_points=1):
    """
    Generic function for detecting relative changes in a given signal.

    Args:
        signal_series: Numpy Array or Pandas Series
        threshold: Float value of a min threshold of values to return as the event
        search_direction: string indicating which direction in the data to begin searching for event, options are forward/backward
        max_theshold: Float value of a max threshold that events have to be under to be an event
        n_points: Number of points in a row meeting threshold criteria to be an event.

    Returns:
        event_idx: Integer of the index where values meet the threshold criteria
    """
    # npoints can't be 0
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
    if max_theshold is not None:
        idx = idx & (arr < max_theshold)
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
        event_idx = ind[0]
    # if 'backward' in search_direction:
    #     event_idx = len(arr) - event_idx - 1

    return event_idx


def get_acceleration_start(acceleration, fractional_basis: float = 0.01, threshold=0.001, max_theshold=0.05):
    """
    Returns the index of the first value that has a relative change
    Args:
        acceleration: np.array or pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold: relative change to indicate start

    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values

    # Find the min value
    ind = np.argwhere((acceleration == acceleration.min()))[0][0]

    # Find the max value of the values before the minimum
    if ind == 0:
        max_ind = 1
    else:
        max_ind = np.argwhere((acceleration[0:ind] == acceleration[0:ind].max()))[0][0]

    # Get the neutral signal between start and the max
    accel_neutral = get_neutral_bias_at_border(acceleration[0:max_ind], fractional_basis=fractional_basis)
    acceleration_start = get_signal_event(accel_neutral, threshold=threshold, max_theshold=max_theshold,
                                          n_points=int(0.01 * len(accel_neutral)),
                                          search_direction='backward')
    return acceleration_start


def get_acceleration_stop(acceleration, fractional_basis=0.01, threshold=-0.001, max_theshold=0.06):
    """
    Returns the index of the last value that has a relative change greater than the
    threshold of absolute normalized signal
    Args:
        acceleration:pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold: Float in g's for

    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    acceleration = acceleration.values
    # Find the min value of the whole array
    ind = np.argwhere((acceleration == acceleration.min()))[0][0]

    # Find the max value between start and the minimum
    if ind == 0:
        max_ind = len(acceleration) - 1
    else:
        max_ind = np.argwhere((acceleration[0:ind] == acceleration[0:ind].max()))[0][0]

    # remove gravity
    accel_neutral = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis)
    # Isolate the area of the signal known to have the stop
    sig = accel_neutral[max_ind:]
    event = get_signal_event(sig, threshold=threshold, max_theshold=max_theshold, n_points=int(0.01 * len(sig)),
                             search_direction='forward')
    if event == 0:
        acceleration_stop = len(acceleration) - 1
    else:
        acceleration_stop = max_ind + event

    return acceleration_stop


def get_nir_surface(ambient, active, fractional_basis=0.01, threshold=0.1):
    """
    Using the active and ambient NIR, estimate the index at when the probe was in the snow.
    The ambient signal is expected to receive less and less light as it enters into the snowpack,
    whereas the active should receive more. Thus this function calculates the first value of the
    difference of the two signals should be the snow surface.

    Args:
        ambient: Numpy Array or pandas Series of the ambient NIR signal
        active: Numpy Array or pandas Series of the active NIR signal
        fractional_basis: Float of begining data to use as the normalization
        threshold: Float threshold value for meeting the snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
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
