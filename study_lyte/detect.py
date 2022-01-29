from numpy import argwhere


def get_signal_event(signal_series, threshold=0.001, search_direction='start'):
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

    if search_direction == 'forward':
        ind = argwhere(sig >= threshold)[0]

    # Handle backwards/backward usage
    elif 'backward' in search_direction:
        ind = argwhere(sig[::-1] >= threshold)[0]
        ind = len(sig) - ind - 1

    else:
        raise ValueError(f'{search_direction} is not a valid event. Use start or end')

    event_idx = ind[0]

    return event_idx


def get_acceleration_start(df, threshold=0.1):
    """
    Return the index of the first values in the
    times series that increase the amount of the threshold.

    Args:
        df: pandas dataframe containing
    """
    pass