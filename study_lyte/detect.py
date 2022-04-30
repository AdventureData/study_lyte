import numpy as np
from .adjustments import get_neutral_bias_at_border

def get_signal_event(signal_series, threshold=0.001, search_direction='forward'):
    """
    Generic function for detecting relative changes in a given signal.

    Args:
        signal_series: Numpy Array or Pandas Series
        threshold: Float value of threshold of values to return as the event
        search_direction: string indicating which direction in the data to begin searching for event, options are forward/backward

    Returns:
        event_idx: Integer of the index where values meet the threshold criteria
    """
    # Parse whether to work with a pandas Series
    if hasattr(signal_series, 'values'):
        sig = signal_series.values
    # Assume Numpy array
    else:
        sig = signal_series

    if 'forward' in search_direction:
        ind = np.argwhere(sig >= threshold)

    # Handle backwards/backward usage
    elif 'backward' in search_direction:
        ind = np.argwhere(sig[::-1] >= threshold)
        ind = len(sig) - ind - 1

    else:
        raise ValueError(f'{search_direction} is not a valid event. Use start or end')

    # If no results are found, return the first index the series
    if len(ind) == 0:
        event_idx = 0
        if 'backward' in search_direction:
            event_idx = len(sig) - event_idx - 1

    else:
        event_idx = ind[0][0]

    return event_idx


def get_acceleration_start(acceleration, fractional_basis: float = 0.01, threshold=0.1):
    """
    Returns the index of the first value that has a relative change
    Args:
        acceleration: np.array or pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold

    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    accel_norm = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis).abs()
    acceleration_start = get_signal_event(accel_norm, threshold=threshold, search_direction='forward')
    return acceleration_start


def get_acceleration_stop(acceleration, fractional_basis=0.01, threshold=0.1):
    """
    Returns the index of the last value that has a relative change greater than the
    threshold of absolute normalized signal
    Args:
        acceleration: np.array or pandas series of acceleration data
        fractional_basis: fraction of the number of points to average over for bias adjustment
        threshold: Float in g's for

    Return:
        acceleration_start: Integer of index in array of the first value meeting the criteria
    """
    accel_norm = get_neutral_bias_at_border(acceleration, fractional_basis=fractional_basis, direction='backwards').abs()
    acceleration_stop = get_signal_event(accel_norm, threshold=threshold, search_direction='backwards')
    return acceleration_stop


def get_nir_surface(ambient, active, n_points_for_basis=1000, threshold=0.1):
    """
    Using the active and ambient NIR, estimate the index at when the probe was in the snow.
    The ambient signal is expected to receive less and less light as it enters into the snowpack,
    whereas the active should receive more. Thus this function calculates the first value of the
    difference of the two signals should be the snow surface.

    Args:
        ambient: Numpy Array or pandas Series of the ambient NIR signal
        active: Numpy Array or pandas Series of the active NIR signal
        n_points_for_basis: Integer Number of points to calculate the mean value used to normalize the signal
        threshold: Float threshold value for meeting the snow surface event

    Return:
        surface: Integer index of the estimated snow surface
    """
    amb_norm = ambient / ambient[0:n_points_for_basis].mean()
    act_norm = active / active[0:n_points_for_basis].mean()

    diff = abs(act_norm - amb_norm)
    surface = get_signal_event(diff, threshold=threshold, search_direction='forward')
    return surface


def get_nir_stop(active, n_points_for_basis=1000, threshold=0.01):
    """
    Often the NIR signal shows the stopping point of the probe by repeated data.
    This looks at the active signal to estimate the stopping point
    """
    bias = active[-1*n_points_for_basis:].min()
    norm = active - bias
    norm = abs(norm / norm.max())
    stop = get_signal_event(norm, threshold=threshold, search_direction='backwards')
    # import matplotlib.pyplot as plt
    # plt.plot(norm)
    # plt.show()
    # print(norm[-10*n_points_for_basis:])
    # print(stop / len(active))
    return stop


def get_acc_maximum(acceleration):
    ind = np.argwhere(acceleration.max())