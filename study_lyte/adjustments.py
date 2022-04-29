import pandas as pd


def get_directional_mean(df: pd.DataFrame, fractional_basis: float = 0.01, direction='forward'):
    """
    Calculates the mean from a collection of points at the beginning or end of a dataframe
    """
    idx = int(fractional_basis * len(df.index))
    if direction == 'forward':
        avg = df.iloc[0:idx].mean()
    elif direction == 'backward':
        avg = df.iloc[-1*idx:].mean()
    else:
        raise ValueError('Invalid Direction used, Use either forward or backward.')
    return avg


def get_neutral_bias_at_border(df: pd.DataFrame, fractional_basis: float = 0.01, direction='forward'):
    """
    Bias adjust the series data by using the XX % of the data either at the front of the data
    or the end of the .
    e.g. 1% of the data is averaged and subtracted.

    Args:
        df: pandas dataframe of data with a known bias
        fractional_basis: Fraction of data to use to estimate the bias on start

    Returns:
        bias_adj: bias adjusted data to near zero
    """
    bias = get_directional_mean(df, fractional_basis=fractional_basis, direction=direction)
    bias_adj = df - bias
    return bias_adj
