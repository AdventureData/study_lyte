from study_lyte.detect import (get_signal_event, get_acceleration_start, get_acceleration_stop, get_nir_surface,
                               get_nir_stop, get_sensor_start, find_nearest_value_index, get_ground_strike)
from study_lyte.io import read_csv
from study_lyte.adjustments import remove_ambient, get_neutral_bias_at_border
import pytest
import numpy as np
import pandas as pd
from os.path import join


@pytest.fixture(scope='session')
def messy_acc(data_dir):
    df, meta = read_csv(join(data_dir, 'messy_acceleration.csv'))
    return df

@pytest.mark.parametrize('value, arr, expected', [
    # Test simple usage
    (10, [0,11,5,4], 1),
    # Tes no real matches found
    (10, [0, 0, 0, 0], 0),
    # Test perfect split picks the lesser of the 2.
    (10, [1, 11.5, 10.5, 0], 2)

])
def test_find_nearest_value_index(value, arr, expected):
    series = pd.Series(arr)
    idx = find_nearest_value_index(value, series)
    assert idx == expected


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
    ([-1, -1, -0.75, -1, -1, 0.5, 1, 0.5, -1, -2, -1.5, -1, -1], 2 / 13, -1.1, 0.5, 4)
])
def test_get_acceleration_start(data, fractional_basis, threshold, max_threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    accel_neutral = get_neutral_bias_at_border(df['acceleration'],
                                               fractional_basis=fractional_basis)
    idx = get_acceleration_start(accel_neutral,
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
    accel_neutral = get_neutral_bias_at_border(raw_df['Y-Axis'])
    idx = get_acceleration_start(accel_neutral)
    assert pytest.approx(idx, abs=int(0.01 * len(raw_df.index))) == start_idx


@pytest.mark.parametrize("data,  fractional_basis, threshold, expected", [
    # Test typical use
    ([-1.0, 1.0, -2, -1.2, -1.1, -0.9, -1.0 -1.0, -1.0], 1 / 9, 0.01, 5),
    # Test no detection returns the last index
    ([-1, -1, -1], 1 / 3, 10, 2),

])
def test_get_acceleration_stop(data, fractional_basis, threshold, expected):
    df = pd.DataFrame({'acceleration': np.array(data)})
    accel_neutral = get_neutral_bias_at_border(df['acceleration'],
                                               fractional_basis=fractional_basis,
                                               direction='backward')
    idx = get_acceleration_stop(accel_neutral, threshold=threshold)

    assert idx == expected


@pytest.mark.parametrize('fname, column, stop_idx', [
    ('messy_acceleration.csv', 'Y-Axis', 353),
    ('bogus.csv', 'Y-Axis', 32112),
    ('fusion.csv', 'Y-Axis', 54083),
    ('kaslo.csv', 'acceleration', 27570),
    ('soft_acceleration.csv', 'Y-Axis', 139),
    ('delayed_acceleration.csv', 'Y-Axis', 266),
    ('gm_data.csv', 'Y-Axis', 8553),
    ('toolik.csv', 'Y-Axis', 17610),
    ('egrip.csv','Y-Axis', 14378),
    ('pilots_error.csv', 'Y-Axis', 12843),
    ('mores_20230119.csv', 'Y-Axis', 18600)

])
def test_get_acceleration_stop_real(raw_df, column, stop_idx):
    accel_neutral = get_neutral_bias_at_border(raw_df[column], direction='backward', fractional_basis=0.005)
    idx = get_acceleration_stop(accel_neutral)
    # Ensure within 3% of original answer all the time.
    assert pytest.approx(idx, abs=int(0.03 * len(raw_df.index))) == stop_idx


@pytest.mark.parametrize('fname', ['fusion.csv'])
def test_get_acceleration_stop_time_index(raw_df):
    """
    Confirm the result is independent of the time index being set or not
    """
    fract = 0.01
    # Without time index
    accel_neutral = get_neutral_bias_at_border(raw_df['Y-Axis'], fractional_basis=fract, direction='backward')
    idx1 = get_acceleration_stop(accel_neutral)
    # with time index
    df = raw_df.set_index('time')
    accel_neutral2 = get_neutral_bias_at_border(df['Y-Axis'], fractional_basis=fract, direction='backward')
    idx2 = get_acceleration_stop(accel_neutral2)

    assert idx1 == idx2


@pytest.mark.parametrize('fname, surface_idx', [
    ('bogus.csv', 20385),
    ('pilots.csv', 9496),
    ('hard_surface_hard_stop.csv', 8515),
    # No Ambient with tester stick
    ('tester_stick.csv', 9887),
    # Noise Ambient
    ('noise_ambient.csv', 14641),
    ('bad_surface_tester.csv', 9583),
    ('toolik.csv', 13684),
    ('banner_legacy.csv', 8177),
    # Get surface with challenging ambient conditions
    ('egrip_tough_surface.csv', 29964),
])
def test_get_nir_surface_real(raw_df, fname, surface_idx):
    """
    Test surface with real data
    """
    # clean = remove_ambient(raw_df['Sensor3'], raw_df['Sensor2'])
    result = get_nir_surface(raw_df['Sensor3'])
    # Ensure within 3% of original answer all the time.
    assert pytest.approx(surface_idx, abs=int(0.02 * len(raw_df.index))) == result

@pytest.mark.parametrize('fname, surface_idx', [
    ('bogus.csv', 32478),
    ('pilots.csv', 24489),
    ('hard_surface_hard_stop.csv', 13501),
    # No Ambient with tester stick
    ('tester_stick.csv', 25273),
    # Noise Ambient
    ('noise_ambient.csv', 23290),

])
def test_get_nir_stop_real(raw_df, fname, surface_idx):
    """
    Test surface with real data
    """
    result = get_nir_stop(raw_df['Sensor3'])
    # Ensure within 3% of original answer all the time.
    assert pytest.approx(surface_idx, abs=int(0.05 * len(raw_df.index))) == result

@pytest.mark.parametrize('fname, column, expected_first_change', [
    ('pilots_error.csv', 'Sensor1', 5758)
])
def test_sensor_start(raw_df, fname, column, expected_first_change):
    first_change = get_sensor_start(raw_df[column])
    assert pytest.approx(first_change, abs=int(0.02 * len(raw_df.index))) == expected_first_change

@pytest.mark.parametrize('fname, expected_ground_strike', [
    # Ground strikes
    ('pilots_error.csv', 12031),
    ('toolik.csv', 17922),
    # Challenging ground strike
    ("ground_touch_and_go.csv", 15389),
    # No ground strikes
    ('egrip_tough_surface.csv', None),
    ('pilots.csv', None),
])
def test_get_ground_strike(raw_df, expected_ground_strike):
    backward_accel = get_neutral_bias_at_border(raw_df['Y-Axis'], direction='backward')
    stop = get_acceleration_stop(backward_accel)
    idx = get_ground_strike(raw_df['Sensor1'], stop)
    assert pytest.approx(idx, abs=int(0.02 * len(raw_df.index))) == expected_ground_strike
