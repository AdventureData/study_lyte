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
    idx = get_signal_event(data, threshold=threshold, search_direction=direction, max_theshold=max_theshold,
                           n_points=n_points)
    assert idx == expected


@pytest.mark.parametrize("data, fractional_basis, threshold, expected", [
    # Test a typical acceleration signal
    ([-1, 0.3, -1.5, -1], 0.25, 0.1, 0),
    ([-1, -1, 1, 0, -2, -1, -1], 0.01, -0.1, 1),
    # # No criteria met, return the first index before the max
    ([-1, -1, -1, -1], 0.25, 10, 0),
    # Test with small bump before start
    ([-1, -1, 0.2, -1, -1, 0.5, 1, 0.5, -1, -2, -1.5, -1, -1], 2 / 13, -1.1, 4)
])
def test_get_acceleration_start(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    idx = get_acceleration_start(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


@pytest.mark.parametrize('fname, start_idx', [
    ('messy_acceleration.csv', 151),
    ('bogus.csv', 17281),
    ('fusion.csv', 32762),
])
def test_get_acceleration_start_messy(raw_df, start_idx):
    idx = get_acceleration_start(raw_df[['Y-Axis']], fractional_basis=0.01, threshold=0.001)
    assert pytest.approx(idx, abs=int(0.01*len(raw_df.index))) == start_idx


@pytest.mark.parametrize("data,  fractional_basis, threshold, expected", [
    # Test typical use
   ([-1.0, 1.0, -2, -1.0, -1.1, -0.9, -1.2], 1 / 7, -0.01, 3),
    # Test a no detection returns the last index
    ([-1, -1, -1], 1 / 3, 10, 2),

])
def test_get_acceleration_stop(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    idx = get_acceleration_stop(df['acceleration'], fractional_basis=fractional_basis, threshold=threshold)
    assert idx == expected


@pytest.mark.parametrize('fname, column, stop_idx', [
    ('messy_acceleration.csv', 'Y-Axis', 262),
    ('bogus.csv', 'Y-Axis', 33342),
    ('fusion.csv', 'Y-Axis', 54083),
    ('kaslo.csv', 'acceleration', 27570),
    ('soft_acceleration.csv', 'Y-Axis', 144)

])
def test_get_acceleration_stop_real(raw_df, column, stop_idx):
    idx = get_acceleration_stop(raw_df[column])
    # import matplotlib.pyplot as plt
    # ax = raw_df[column].plot()
    # ax.axvline(raw_df.index[idx])
    # plt.show()
    # Ensure within 1% of original answer all the time.
    assert pytest.approx(idx, abs=int(0.01*len(raw_df.index))) == stop_idx


@pytest.mark.parametrize('fname', ['fusion.csv'])
def test_get_acceleration_stop_time_index(raw_df):
    """
    Confirm the result is independent of the time index being set or not
    """
    # Without time index
    idx1 = get_acceleration_stop(raw_df['Y-Axis'], fractional_basis=0.01, threshold=-0.001)
    # with time index
    df = raw_df.set_index('time')
    idx2 = get_acceleration_stop(df['Y-Axis'], fractional_basis=0.01, threshold=-0.001)

    assert idx1 == idx2


@pytest.mark.parametrize("ambient, active, fractional_basis, threashold, expected", [
    # Typical bright->dark ambient
    ([3000, 3000, 1000, 100], [2500, 2500, 3000, 4000], 0.25, 0.1, 2),
    # no ambient change ( dark or super cloudy)
    ([100, 100, 100, 100], [1000, 1000, 1000, 2000], 0.5, 0.1, 3),
    # 1/2 split using defaults
    ([1, 1, 2, 2], [2, 2, 1, 1], 0.01, 0.1, 2)
])
def test_get_nir_surface(ambient, active, fractional_basis, threashold, expected):
    df = pd.DataFrame({'ambient': np.array(ambient),
                       'active': np.array(active)})

    idx = get_nir_surface(df['ambient'], df['active'], fractional_basis=fractional_basis, threshold=threashold)
    assert idx == expected


@pytest.mark.parametrize('fname, surface_idx', [
    ('bogus.csv', 20385),
])
def test_get_nir_surface_real(raw_df, fname, surface_idx):
    """
    Test surface with real data
    """
    result = get_nir_surface(raw_df['Sensor3'], raw_df['Sensor2'])
    assert result == surface_idx