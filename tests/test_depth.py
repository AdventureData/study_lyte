from study_lyte.depth import get_depth_from_acceleration
from study_lyte.io import read_csv
import pytest
import numpy as np
from os.path import join

import matplotlib.pyplot as plt

@pytest.fixture()
def accel(data_dir):
    df, meta = read_csv(join(data_dir, 'raw_acceleration.csv'))
    cols = [c for c in df.columns if 'Axis' in c]
    df[cols] = df[cols].mul(2)

    return df

def test_get_depth_from_acceleration(accel):
    """
    Test our ability to extract position of the probe from acceleration
    """
    depth = get_depth_from_acceleration(accel, return_axis='Y-Axis')
    plt.plot(accel['time'], depth)
    plt.show()
    np.testing.assert_equal(depth, expected_depth)