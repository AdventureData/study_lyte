import pytest
from os.path import dirname, join


@pytest.fixture(scope='session')
def data_dir():
    return join(dirname(__file__), 'data')
