import numpy as np
from statistics import NormalDist


def z_score(confidence):
    center_prob = (1 + confidence) / 2.
    z = NormalDist().inv_cdf(center_prob)
    return z


def margin_of_error(n, std, confidence=0.95):
    """
    Calculate the margin of error without scipy.
    """
    z = z_score(confidence)
    moe = z * np.sqrt(std**2 / n)
    return moe


def required_sample_for_margin(desired_margin_of_error, std, confidence=0.95):
    """
    Calculate the required sample size for desired margin of error without scipy.
    """
    z = z_score(confidence)
    n = (std ** 2) / ((desired_margin_of_error / z) ** 2)
    return n