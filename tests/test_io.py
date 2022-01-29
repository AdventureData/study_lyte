from study_lyte.io import read_csv
import pytest
from os.path import join

@pytest.mark.parametrize("f, expected_columns", [
    ('hi_res.csv', ['Unnamed: 0', 'Sensor1', 'Sensor2', 'Sensor3', 'acceleration', 'depth']),
    ('rad_app.csv', ['SAMPLE', 'SENSOR 1', 'SENSOR 2', 'SENSOR 3', 'SENSOR 4', 'DEPTH'])
])
def test_read_csv(data_dir, f, expected_columns):
    """
    Test the read_csv function
    """
    df, meta = read_csv(join(data_dir, f))
    print(df.columns)
    assert sorted(df.columns) == sorted(expected_columns)
