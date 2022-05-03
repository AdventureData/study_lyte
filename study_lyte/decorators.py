import functools

import pandas as pd


def time_series(func):
    """
    Decorator to use for functions that require a time index.
    Checks if time is used as the index or is in the columns. If it
    is in the column, make it an index.Otherwise throw an error.
    """

    @functools.wraps(func)
    def set_time_series(pd_obj, *args, **kwargs):
        if type(pd_obj) == pd.DataFrame:
            if pd_obj.index.name != 'time' and 'time' not in pd_obj.columns:
                raise ValueError(f"Time series data requires a 'time' column or index named time to calculate!")
            elif pd_obj.index.name != 'time' and 'time' in pd_obj.columns:
                pd_obj = pd_obj.set_index('time')

        elif type(pd_obj) == pd.Series:
            if pd_obj.index.name != 'time':
                raise ValueError(f"Time series data requires index named 'time' to calculate!")

        result = func(pd_obj, *args, **kwargs)
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

