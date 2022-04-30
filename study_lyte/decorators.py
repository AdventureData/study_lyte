import functools


def time_series(func):
    """
    Decorator to use for functions that require a time index.
    Checks if time is used as the index or is in the columns. If it
    is in the column, make it an index.Otherwise throw an error.
    """

    @functools.wraps(func)
    def set_time_series(df, *args, **kwargs):
        if df.index.name != 'time' and 'time' not in df.columns:
            raise ValueError(f"Time series data requires a 'time' column or index named time to calculate!")

        if 'time' in df.columns:
            df = df.set_index('time')
        result = func(df, *args, **kwargs)

        return result

    return set_time_series


def directional(_func=None, *, check='direction'):
    """
    Decorator to check if the direction specified is valid, use this to
    standardize all directions and value checking
    """
    def decorator_directional(func):
        @functools.wraps(func)
        def check_directionality(*args, **kwargs):
            if kwargs[check] not in ['forward', 'backward']:
                raise ValueError(f'{check} = {kwargs[check]} is an invalid direction, use either forward or backward.')

            result = func(*args, **kwargs)
            return result
        return check_directionality

    if _func is None:
        return decorator_directional
    else:
        return decorator_directional(_func)

