from study_lyte.detect import get_signal_event, get_acceleration_start, get_acceleration_stop, get_nir_surface
import pytest
import numpy as np
import pandas as pd


@pytest.mark.parametrize("data, threshold, direction, expected", [
    (np.array([0, 1, 0]), 0.5, 'forward', 1),
    (pd.Series([1, 0, 0]), 0.5, 'forward', 0),
    (np.array([0, 0, 0.1]), 0.01, 'backwards', 2),
    (np.array([20, 100, 100, 40]), 50, 'backwards', 2),
])
def test_get_signal_event(data, threshold, direction, expected):
    """
    Test the signal event is capable of return the correct index
    regardless of threshold, direction, and series or numpy array
    """
    idx = get_signal_event(data, threshold=threshold, search_direction=direction)
    assert idx == expected


@pytest.mark.parametrize("data,  n_points_for_basis, threshold, expected", [
    # Test normalization and abs
    ([-1, -0.5, -0.1], 1, 0.1, 1),
    # Test the mean basis value
    ([-1, -3, -1, -2], 2, 1, 3),
])
def test_get_acceleration_start(data, n_points_for_basis, threshold, expected):
    d = np.array(data)
    idx = get_acceleration_start(d, n_points_for_basis=n_points_for_basis, threshold=threshold)


@pytest.mark.parametrize("data,  n_points_for_basis, threshold, expected", [
    # Test normalization and abs
    ([0.1, -1.5, -1.0], 1, 1.0, 0),
])
def test_get_acceleration_stop(data, n_points_for_basis, threshold, expected):
    d = np.array(data)
    idx = get_acceleration_stop(d, n_points_for_basis=n_points_for_basis, threshold=threshold)


@pytest.mark.parametrize("ambient, active, n_points_for_basis, threashold, expected", [
    # Typical bright->dark ambient
    ([3000, 3000, 1000, 100], [2500, 2500, 3000, 4000], 1, 0.1, 2),
    # no ambient change ( dark or super cloudy)
    ([100, 100, 100, 100], [1000, 1000, 1000, 2000], 1, 0.1, 3)

])
def test_get_nir_surface(ambient, active, n_points_for_basis, threashold, expected):
    amb = np.array(ambient)
    act = np.array(active)
    idx = get_nir_surface(amb, act, n_points_for_basis=n_points_for_basis, threshold=threashold)
    assert idx == expected
