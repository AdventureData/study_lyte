import pandas as pd
from os.path import join
import pytest

from study_lyte.io import read_csv
from study_lyte.detect import get_acceleration_start, get_acceleration_stop
from study_lyte.depth import get_depth_from_acceleration, get_fitted_depth, \
    get_constrained_baro_depth
from study_lyte.adjustments import assume_no_upward_motion, get_neutral_bias_at_border

@pytest.fixture(scope='session')
def accel(data_dir):
    """
    Real accelerometer data
    """
    df, meta = read_csv(join(data_dir, 'raw_acceleration.csv'))
    cols = [c for c in df.columns if 'Axis' in c]
    df[cols] = df[cols].mul(2)

    return df


@pytest.fixture(scope='session')
def unfiltered_baro(data_dir):
    """
    Real accelerometer data
    """
    df, meta = read_csv(join(data_dir, 'unfiltered_baro.csv'))
    return df


@pytest.mark.parametrize('component, expected_delta', [
    ('X-Axis', 27.0),
    ('Y-Axis', 51.6),
    ('Z-Axis', 65.8),
    ('magnitude', 83.6)])
def test_get_depth_from_acceleration_full(accel, component, expected_delta):
    """
    Test extracting position of the probe from acceleration on real data
    """
    neutral = accel.apply(lambda col: get_neutral_bias_at_border(col), axis=0)
    depth = get_depth_from_acceleration(neutral)
    delta = depth.max() - depth.min()
    assert pytest.approx(delta[component], abs=0.1) == expected_delta


def test_get_depth_from_acceleration_partial(accel):
    """
    Test magnitude is not calculated when not all the columns are present
    """
    depth = get_depth_from_acceleration(accel[['time', 'Y-Axis']])
    assert 'magnitude' not in depth.columns


def test_get_depth_from_acceleration_full_exception(accel):
    """
    Test raising an error on no time column or index
    """
    with pytest.raises(ValueError):
        df = get_depth_from_acceleration(accel.reset_index().drop(columns='time'))


def test_get_fitted_depth(unfiltered_baro):
    df = get_fitted_depth(unfiltered_baro, column='filtereddepth', poly_deg=5)
    delta = df['fitted_filtereddepth'].max() - df['fitted_filtereddepth'].min()
    assert pytest.approx(delta, abs=5) == 130.77156406716017


@pytest.mark.parametrize('depth_data, acc_data, start, stop, expected', [
    # Simple example where peak/valley is beyond start/stop
    ([1.0, 1.2, 0.9, 0.75, 0.5, 0.25, 0.1, -0.2, 0], [-1, -1, -0.98, -0.01, 1, -2, -1, -1, -1], 2, 6, 1.1),
    # Confirm avg of tails and rescale
    ([1.1, 0.9, 1.5, 1.0, 0.5, 0, -0.5, -0.1, 0.1], [-1, -1, -1, -0.98, 1.5, -1, -1,  -1, -1], 3, 6., 1.2),
    # Example with no peak valley found
    ([1.5, 1.0, 0.5, 0, -0.5], [-1, -0.98, 1.5, -2.5, -1.1], 1, 4., 2.0)

])
def test_get_constrained_baro_depth(depth_data, acc_data, start, stop, expected):
    t = range(len(depth_data))
    df = pd.DataFrame.from_dict({'depth': depth_data, 'Y-Axis': acc_data, 'time': t})
    neutral = get_neutral_bias_at_border(df['Y-Axis'])
    start = get_acceleration_start(neutral)
    stop = get_acceleration_stop(neutral)

    result = get_constrained_baro_depth(df.set_index('time')['depth'], start, stop)
    result_s = result.index[0]
    result_e = result.index[-1]
    delta_h_result = result.iloc[0] - result.iloc[-1]
    assert (result_s, result_e, pytest.approx(abs(delta_h_result), abs=0.01)) == (start, stop, expected)


@pytest.mark.parametrize('fname, column, acc_axis, method, expected_depth', [
    ('hard_surface_hard_stop.csv', 'depth', 'Y-Axis', 'nanmean', 90),
    ('baro_w_bench.csv', 'filtereddepth', 'Y-Axis', 'nanmedian', 43),
    ('baro_w_tails.csv', 'filtereddepth', 'Y-Axis', 'nanmean', 62),
    ('smooth.csv', 'filtereddepth', 'Y-Axis', 'nanmedian',  65),
    ('low_zpfo_baro.csv', 'filtereddepth', 'Y-Axis', 'nanmedian', 65),
    ('lower_slow_down.csv', 'filtereddepth', 'Y-Axis', 'nanmedian', 55),
    ('pilots.csv', 'depth', 'Y-Axis', 'nanmedian', 211),
    ('mores_pit_1.csv', 'depth', 'Y-Axis', 'nanmedian', 130),
    ('rough_bench.csv', 'filtereddepth', 'Y-Axis', 'nanmedian', 52),
])
def test_get_constrained_baro_real(raw_df, fname, column, acc_axis, method, expected_depth):
    """
    Test the constrained baro with acceleration data
    """
    neutral = get_neutral_bias_at_border(raw_df[acc_axis])
    start = get_acceleration_start(neutral)
    stop = get_acceleration_stop(neutral)
    result = get_constrained_baro_depth(raw_df.set_index('time')[column], start, stop, method=method)
    delta_d = abs(result.max() - result.min())
    assert pytest.approx(delta_d, abs=3) == expected_depth
