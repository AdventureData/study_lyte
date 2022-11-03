import pandas as pd
from os.path import join
import pytest
import numpy as np

from study_lyte.io import read_csv
from study_lyte.depth import get_depth_from_acceleration, get_average_depth, get_fitted_depth, \
    get_constrained_baro_depth


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
    ('X-Axis', 0.2699937),
    ('Y-Axis', 0.5155876),
    ('Z-Axis', 0.6578470),
    ('magnitude', 0.8362483)])
def test_get_depth_from_acceleration_full(accel, component, expected_delta):
    """
    Test extracting position of the probe from acceleration on real data
    """
    depth = get_depth_from_acceleration(accel)
    delta = depth.max() - depth.min()
    assert pytest.approx(delta[component], abs=1e-6) == expected_delta


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


def test_get_average_depth(peripherals):
    result = get_average_depth(peripherals, depth_column='filtereddepth')
    delta = result['depth'].max() - result['depth'].min()
    assert pytest.approx(delta, abs=1e-6) == 48.9902805


def test_get_fitted_depth(unfiltered_baro):
    df = get_fitted_depth(unfiltered_baro, column='filtereddepth', poly_deg=5)
    delta = df['fitted_filtereddepth'].max() - df['fitted_filtereddepth'].min()
    assert pytest.approx(delta, abs=5) == 130.77156406716017

@pytest.mark.parametrize('depth_data, acc_data', [
    # Simple example where peak/valley is beyond start/stop
    ([1, 0.75, 0.5, 0.25, 0], [-1, 1, 0, -2, -1])
])
def test_get_constrained_baro_depth(depth_data, acc_data):
    t = range(len(depth_data))
    df = pd.DataFrame.from_dict({'depth': depth_data, 'Y-Axis': acc_data, 'time': t})
    result = get_constrained_baro_depth(df)

@pytest.mark.parametrize('fname, column, expected_depth', [
    ('fusion.csv', 'depth', 104),
    ('bogus.csv', 'depth', 127),
])
def test_get_constrained_baro(raw_df, fname, column, expected_depth):
    df = get_constrained_baro_depth(raw_df, baro=column)
    delta_d = abs(df[column].max() - df[column].min())
    assert pytest.approx(delta_d, abs=2) == expected_depth

@pytest.mark.parametrize('fname, depth_column, acc_column, expected_depth', [
    ('/home/micah/Dropbox/RAD_Company/case_studies/kaslo_02-13-2022/2022-02-13--123149.csv', 'depth', 'acceleration', 100),
    ('/home/micah/Dropbox/RAD_Company/case_studies/kaslo_02-13-2022/2022-02-13--123233.csv', 'depth',
     'acceleration', 100)

])
def test_baro_constraint_real(fname, depth_column, acc_column, expected_depth):
    df, meta = read_csv(fname)
    df['time'] = df.index / 16000
    result = get_constrained_baro_depth(df, baro=depth_column, acc_axis=acc_column)