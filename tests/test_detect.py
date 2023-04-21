from study_lyte.detect import get_signal_event, get_acceleration_start, get_acceleration_stop, get_nir_surface
from study_lyte.io import read_csv
from study_lyte.adjustments import remove_ambient
import pytest
import numpy as np
import pandas as pd
from os.path import join


@pytest.fixture(scope='session')
def messy_acc(data_dir):
    df, meta = read_csv(join(data_dir, 'messy_acceleration.csv'))
    return df


@pytest.mark.parametrize("data, threshold, direction, max_threshold, n_points, expected", [
    (np.array([0, 1, 0]), 0.5, 'forward', None, 1, 1),
    (pd.Series([1, 0, 0]), 0.5, 'forward', None, 1, 0),
    (np.array([0, 0, 0.1]), 0.01, 'backward', None, 1, 2),
    (np.array([20, 100, 100, 40]), 50, 'backward', None, 1, 1),
    # Test max threshold + threshold
    (np.array([20, 100, 100, 40]), 30, 'forward', 90, 1, 3),
    # n_points test
    (np.array([2, 2, 1, 2]), 2, 'forward', None, 2, 1),
    # All together
    (np.array([11, 10, 1, 2, 2, 3, 11]), 2, 'forward', 10, 3, 5),
])
def test_get_signal_event(data, threshold, direction, max_threshold, n_points, expected):
    """
    Test the signal event is capable of return the correct index
    regardless of threshold, direction, and series or numpy array
    """
    idx = get_signal_event(data, threshold=threshold,
                           search_direction=direction,
                           max_threshold=max_threshold,
                           n_points=n_points)
    assert idx == expected


@pytest.mark.parametrize("data, fractional_basis, threshold, max_threshold, expected", [
    # Test a typical acceleration signal with a no zero start
    ([-1, -1, 0.3, -1.5, -1], 0.2, 0, 0.5, 1),
    ([-1, -1, -0.8, 1, 0, -2, -1, -1], 0.01, 0.1, 0.3, 2),
    # No criteria met, return the first index before the max
    ([-1, -1, -1, -1], 0.25, 10, 11, 0),
    # Test with small bump before start
    ([-1, -1, 0.05, -1, -1, 0.5, 1, 0.5, -1, -2, -1.5, -1, -1], 2 / 13, -1.1, 0.5, 4)
])
def test_get_acceleration_start(data, fractional_basis, threshold, max_threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    idx = get_acceleration_start(df['acceleration'],
                                 fractional_basis=fractional_basis,
                                 threshold=threshold,
                                 max_threshold=max_threshold)
    assert idx == expected


@pytest.mark.parametrize('fname, start_idx', [
    ('messy_acceleration.csv', 37),
    ('bogus.csv', 16178),
    ('fusion.csv', 32762),
    ('delayed_acceleration.csv', 160),
    ('hard_surface_hard_stop.csv', 5057)
])
def test_get_acceleration_start_messy(raw_df, start_idx):
    idx = get_acceleration_start(raw_df[['Y-Axis']])
    assert pytest.approx(idx, abs=int(0.01 * len(raw_df.index))) == start_idx


@pytest.mark.parametrize("data,  fractional_basis, threshold, expected", [
    # Test typical use
    ([-1.0, 1.0, -2, -1.0, -1.1, -0.9, -1.2], 1 / 7, 0.01, 3),
    # Test no detection returns the last index
    ([-1, -1, -1], 1 / 3, 10, 2),

])
def test_get_acceleration_stop(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    idx = get_acceleration_stop(df['acceleration'],
                                fractional_basis=fractional_basis,
                                height=threshold)

    assert idx == expected


@pytest.mark.parametrize('fname, column, stop_idx', [
    ('messy_acceleration.csv', 'Y-Axis', 353),
    ('bogus.csv', 'Y-Axis', 32112),
    ('fusion.csv', 'Y-Axis', 54083),
    ('kaslo.csv', 'acceleration', 27570),
    ('soft_acceleration.csv', 'Y-Axis', 131),
    ('delayed_acceleration.csv', 'Y-Axis', 252)
])
def test_get_acceleration_stop_real(raw_df, column, stop_idx):
    idx = get_acceleration_stop(raw_df[column])
    # Ensure within 3% of original answer all the time.
    assert pytest.approx(idx, abs=int(0.03 * len(raw_df.index))) == stop_idx


@pytest.mark.parametrize('fname', ['fusion.csv'])
def test_get_acceleration_stop_time_index(raw_df):
    """
    Confirm the result is independent of the time index being set or not
    """
    # Without time index
    idx1 = get_acceleration_stop(raw_df['Y-Axis'], fractional_basis=0.01)
    # with time index
    df = raw_df.set_index('time')
    idx2 = get_acceleration_stop(df['Y-Axis'], fractional_basis=0.01)

    assert idx1 == idx2


@pytest.mark.parametrize("active, threshold, max_threshold, expected", [
    # Typical bright->dark ambient
    ([0, 200, 3000, 4000], 0.01, 0.1, 1),
    # no ambient change ( dark or super cloudy)
    ([1000, 1100, 2000, 3000], .01, 0.2,  1),
    # 1/2 split using defaults
    ([0, 1, 3, 5], 0.5, 0.7, 2)

])
def test_get_nir_surface(active, threshold, max_threshold, expected):
    idx = get_nir_surface(np.array(active),
                          threshold=threshold,
                          max_threshold=max_threshold)
    assert idx == expected


@pytest.mark.parametrize('fname, surface_idx', [
    ('bogus.csv', 20385),
    ('pilots.csv', 9496),
    ('hard_surface_hard_stop.csv', 10167),
    # No Ambient with tester stick
    ('tester_stick.csv', 9887),
    # Noise Ambient
    ('noise_ambient.csv', 14641),
])
def test_get_nir_surface_real(raw_df, fname, surface_idx):
    """
    Test surface with real data
    """
    clean = remove_ambient(raw_df['Sensor3'], raw_df['Sensor2'])

    result = get_nir_surface(clean)
    # Ensure within 3% of original answer all the time.
    assert pytest.approx(surface_idx, abs=int(0.02 * len(raw_df.index))) == result
