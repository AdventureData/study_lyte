from study_lyte.io import (MEASUREMENT_ID_KEY, find_measurement_id,
                           new_measurement_id, read_csv, write_csv)
from uuid import UUID
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


class TestMeasurementId:
    """
    The id that lets a measurement be named.

    Before it, the only key was (Serial Num., RECORDED), which collides inside
    a second and has no timezone. Three clients write these files
    independently, so the tolerance below is not politeness — it is what stops
    a stray underscore silently orphaning a backup.
    """

    def test_new_ids_are_unique(self):
        assert len({new_measurement_id() for _ in range(100)}) == 100

    def test_new_ids_are_uuid4(self):
        assert UUID(new_measurement_id()).version == 4

    @pytest.mark.parametrize('key', [
        'MEASUREMENT ID',
        'MEASUREMENT_ID',
        'measurement id',
        'measurement_id',
        'Measurement Id',
    ])
    def test_key_spellings_all_resolve(self, key):
        value = new_measurement_id()

        assert find_measurement_id({key: value}) == value

    @pytest.mark.parametrize('metadata', [
        {},
        {'RECORDED': '2026-07-20--21:44:55'},
        # Present but empty is the same as absent, not an id of ''
        {'MEASUREMENT ID': ''},
        {'MEASUREMENT ID': '   '},
    ])
    def test_absent_id_is_none(self, metadata):
        assert find_measurement_id(metadata) is None

    def test_value_is_stripped(self):
        value = new_measurement_id()

        assert find_measurement_id({'MEASUREMENT ID': f'  {value}\t'}) == value

    def test_survives_a_write_and_read(self, tmp_path):
        """The id has to come back out of a real file, not just a dict."""
        value = new_measurement_id()
        out = tmp_path / 'measurement.csv'

        write_csv(DataFrame({'depth': [0.0, -1.0], 'Sensor1': [1, 2]}),
                  {MEASUREMENT_ID_KEY: value, 'Serial Num.': 'ABCD100A0E010001'},
                  str(out))

        df, meta = read_csv(str(out))

        assert find_measurement_id(meta) == value
        assert sorted(df.columns) == ['Sensor1', 'depth']
