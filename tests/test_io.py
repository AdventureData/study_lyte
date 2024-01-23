from study_lyte.io import read_csv, write_csv
import pytest
from os.path import join, isfile
import os
from pandas import DataFrame


@pytest.mark.parametrize("f, expected_columns", [
    ('hi_res.csv', ['Sensor1', 'Sensor2', 'Sensor3', 'acceleration', 'depth', 'time']),
    ('rad_app.csv', ['SAMPLE', 'SENSOR 1', 'SENSOR 2', 'SENSOR 3', 'SENSOR 4', 'DEPTH'])
])
def test_read_csv_columns(data_dir, f, expected_columns):
    """
    Test the read_csv function
    """
    df, meta = read_csv(join(data_dir, f))
    assert sorted(df.columns) == sorted(expected_columns)


@pytest.mark.parametrize("f, expected_meta", [
    ('hi_res.csv', {"RECORDED": "2022-01-23--11:30:16",
                    "radicl VERSION": "0.5.1",
                    "FIRMWARE REVISION": "1.46",
                    "HARDWARE REVISION": '1',
                    "MODEL NUMBER": "3",
                    "SAMPLE RATE": "16000"}),
    ('rad_app.csv', {"LOCATION": "43.566, -116.121",
                     "APP REVISION": "1.17.1",
                     "MODEL_NUMBER": "PB2",
                     "PROCESSING ALGORITHM": "2"})
])
def test_read_csv_meta(data_dir, f, expected_meta):
    """
    Test the read_csv function
    """
    df, meta = read_csv(join(data_dir, f))
    assert meta == expected_meta


@pytest.fixture()
def out_file():
    f = 'test_output.csv'
    yield f
    if isfile(f):
        os.remove(f)


def test_write_csv(out_file):
    """
    Test the writing of a csv with metadata
    """
    meta = {"model": "10"}
    df = DataFrame({'data': [1, 2, 3]})
    write_csv(df, meta, out_file)

    with open(out_file) as fp:
        txt = ''.join(fp.readlines())
    assert txt == 'model = 10\ndata\n1\n2\n3\n'
