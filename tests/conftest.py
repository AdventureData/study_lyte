import pytest
from os.path import dirname, join
from study_lyte.io import read_csv


@pytest.fixture(scope='session')
def data_dir():
    return join(dirname(__file__), 'data')


@pytest.fixture(scope='function')
def raw_df(data_dir, fname):
    df, meta = read_csv(join(data_dir, fname))
    return df


@pytest.fixture(scope='session')
def peripherals(data_dir):
    """
    Return a df of accelerometer and filtered barometer data
    """
    df, meta = read_csv(join(data_dir, 'peripherals.csv'))
    return df
