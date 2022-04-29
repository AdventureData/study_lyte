import functools

def time_series(func):
    """
    Decorator to use for functions that bump the state along
    for the state machine in rummager
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
