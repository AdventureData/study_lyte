from study_lyte.depth import get_depth_from_acceleration
from study_lyte.io import read_csv
import pytest
from os.path import join


@pytest.fixture(scope='session')
def accel(data_dir):
    """
    Real accelerometer data
    """
    df, meta = read_csv(join(data_dir, 'raw_acceleration.csv'))
    cols = [c for c in df.columns if 'Axis' in c]
    df[cols] = df[cols].mul(2)

    return df


@pytest.mark.parametrize('component, expected_delta', [
    ('X-Axis', 0.2692459),
    ('Y-Axis', 0.510274),
    ('Z-Axis', 0.411576),
    ('magnitude', 0.634314)])
def test_get_depth_from_acceleration_full(accel, component, expected_delta):
    """
    Test extracting position of the probe from acceleration on real data
    """
    depth = get_depth_from_acceleration(accel)
    delta = depth.max() - depth.min()
    assert pytest.approx(delta[component], abs=1e-6) == expected_delta


def test_get_depth_from_acceleration_full_exception(accel):
    """
    Test raising an error on no time column or index
    """
    with pytest.raises(ValueError):
        get_depth_from_acceleration(accel.reset_index().drop(columns='time'))
