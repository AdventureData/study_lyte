import pytest
from os.path import dirname, join
from study_lyte.io import read_csv
from study_lyte.profile import LyteProfileV6
from pathlib import Path

@pytest.fixture(scope='session')
def data_dir():
    return Path(__file__).parent.joinpath('data')


@pytest.fixture(scope='function')
def raw_df(data_dir, fname):
    df, meta = read_csv(data_dir.joinpath(fname))
    return df

@pytest.fixture(scope='function')
def lyte_profile(data_dir, fname):
    return LyteProfileV6(join(data_dir, fname), calibration={'Sensor1': [1, 0]})


@pytest.fixture(scope='session')
def peripherals(data_dir):
    """
    Return a df of accelerometer and filtered barometer data
    """
    df, meta = read_csv(join(data_dir, 'peripherals.csv'))
    return df
