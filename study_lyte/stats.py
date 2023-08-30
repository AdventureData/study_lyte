import numpy as np
import scipy.stats as st

def margin_of_error(n, std, confidence=0.95):
    """
    Calculate the margin of error
    Args:
        n: sample size
        std: Standard deviation
        confidence: fraction for confidence interval
    Returns:
        moe: float of the margin of error
    """
    z = st.norm.ppf(confidence)
    moe = z * np.sqrt(std**2 / n)
    return moe

def required_sample_for_margin( desired_margin_of_error, std, confidence=0.95):
    """
    Calculate the required sample size for desired margin of error

    Args:
        desired_margin_of_error: Standard deviation
        std: Standard deviation
        confidence: fraction for confidence interval

    Returns:
        moe: float of the margin of error
    """
    z = st.norm.ppf(confidence)
    n = (std ** 2)/ ((desired_margin_of_error / z)**2)
    return n