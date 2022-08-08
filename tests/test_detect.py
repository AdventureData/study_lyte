from study_lyte.detect import get_signal_event, get_acceleration_start, get_acceleration_stop, get_nir_surface
from study_lyte.io import read_csv
import pytest
import numpy as np
import pandas as pd
from os.path import join


@pytest.fixture(scope='session')
def messy_acc(data_dir):
    df, meta = read_csv(join(data_dir, 'messy_acceleration.csv'))
    return df


@pytest.mark.parametrize("data, threshold, direction, max_theshold, n_points, expected", [
    (np.array([0, 1, 0]), 0.5, 'forward', None, 1, 1),
    (pd.Series([1, 0, 0]), 0.5, 'forward', None, 1, 0),
    (np.array([0, 0, 0.1]), 0.01, 'backward', None, 1, 2),
    (np.array([20, 100, 100, 40]), 50, 'backward', None, 1, 2),
    # Test max threshold + threshold
    (np.array([20, 100, 100, 40]), 30, 'forward', 90, 1, 3),
    # n_points test
    (np.array([2, 2, 1, 2]), 2, 'forward', None, 2, 1),
    # All togehter
    (np.array([11, 10, 1, 2, 2, 3, 11]), 2, 'forward', 10, 3, 5),
])
def test_get_signal_event(data, threshold, direction, max_theshold, n_points, expected):
    """
    Test the signal event is capable of return the correct index
    regardless of threshold, direction, and series or numpy array
    """
    idx = get_signal_event(data, threshold=threshold, search_direction=direction, max_theshold=max_theshold, n_points=n_points)
    assert idx == expected


@pytest.mark.parametrize("data, fractional_basis, threshold, expected", [
    # Test a typical acceleration signal
    ([-1, 0.3, -1.5, -1], 0.25, 0.1, 0),
    # No criteria met, return the first index before the max
    ([-1, -1, -1, -1], 0.25, 10, 0),
    # Test with small bump before start
    ([-1, -1, 0.2, -1, -1, 0.5, 1, 0.5, -1, -2, -1.5, -1, -1], 2/13, 0.1, 5)
])
def test_get_acceleration_start(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration':  np.array(data)})
    idx = get_acceleration_start(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


def test_get_acceleration_start_messy(messy_acc):
    idx = get_acceleration_start(messy_acc[['Y-Axis']])
    assert idx == 144


@pytest.mark.parametrize("data,  fractional_basis, threshold, expected", [
    # Test typical use
    ([-1.0, 1.0, -2, -1.0, -1.1, -0.9, -1.2], 1/7, -0.01, 3),
    # Test a no detection returns the last index
    ([-1, -1, -1], 1 / 3, 10, 2),
])
def test_get_acceleration_stop(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration':  np.array(data)})
    idx = get_acceleration_stop(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


def test_get_acceleration_stop_messy(messy_acc):
    idx = get_acceleration_stop(messy_acc[['Y-Axis']])
    assert idx == 273


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
