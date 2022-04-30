from study_lyte.detect import get_signal_event, get_acceleration_start, get_acceleration_stop, get_nir_surface
import pytest
import numpy as np
import pandas as pd


@pytest.mark.parametrize("data, threshold, direction, expected", [
    (np.array([0, 1, 0]), 0.5, 'forward', 1),
    (pd.Series([1, 0, 0]), 0.5, 'forward', 0),
    (np.array([0, 0, 0.1]), 0.01, 'backward', 2),
    (np.array([20, 100, 100, 40]), 50, 'backward', 2),
])
def test_get_signal_event(data, threshold, direction, expected):
    """
    Test the signal event is capable of return the correct index
    regardless of threshold, direction, and series or numpy array
    """
    idx = get_signal_event(data, threshold=threshold, search_direction=direction)
    assert idx == expected


@pytest.mark.parametrize("data, fractional_basis, threshold, expected", [
    # Test a typical acceleration signal
    ([-1, 0.3, -1.5, -1], 0.25, 0.1, 1),
    # Test the mean basis value
    ([-2, -6, -8, -12], 0.5, 2.1, 2),
    # No criteria met, return the first index
    ([-1, -1, -1, -1], 0.25, 10, 0),
])
def test_get_acceleration_start(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration':  np.array(data)})
    idx = get_acceleration_start(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


@pytest.mark.parametrize("data,  fractional_basis, threshold, expected", [
    # Test normalization and abs
    ([0.1, -1.5, -1.0], 1 / 3, 1.0, 0),
    # Test a no detection returns the last index
    ([-1, -1, -1], 1 / 3, 10, 2),
])
def test_get_acceleration_stop(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration':  np.array(data)})
    idx = get_acceleration_stop(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


@pytest.mark.parametrize("ambient, active, fractional_basis, threashold, expected", [
    # Typical bright->dark ambient
    ([3000, 3000, 1000, 100], [2500, 2500, 3000, 4000], 0.25, 0.1, 2),
    # no ambient change ( dark or super cloudy)
    ([100, 100, 100, 100], [1000, 1000, 1000, 2000], 0.5, 0.1, 3)

])
def test_get_nir_surface(ambient, active, fractional_basis, threashold, expected):
    df = pd.DataFrame({'ambient':  np.array(ambient),
                       'active':  np.array(active)})

    idx = get_nir_surface(df['ambient'], df['active'], fractional_basis=fractional_basis, threshold=threashold)
    assert idx == expected
