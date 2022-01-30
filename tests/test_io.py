from study_lyte.io import read_csv
import pytest
from os.path import join


@pytest.mark.parametrize("f, expected_columns", [
    ('hi_res.csv', ['Unnamed: 0', 'Sensor1', 'Sensor2', 'Sensor3', 'acceleration', 'depth']),
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
                    "MODEL NUMBER": "3"}),
    ('rad_app.csv', {"LOCATION": "43.566052, -116.12157452",
                     "APP REVISION": "1.17.1",
                     "MODEL_NUMBER": "PB2",
                     "PROCESSING ALGORITHM": "2"})
])
def test_read_csv_meta(data_dir, f, expected_meta):
    """
    Test the read_csv function
    """
    df, meta = read_csv(join(data_dir, f))
    assert meta == pytest.approx(expected_meta)
