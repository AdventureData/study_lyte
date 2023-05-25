import pytest
from os.path import join
from study_lyte.profile import LyteProfileV6, Event


class TestLyteProfile:

    @pytest.fixture()
    def profile(self, data_dir, filename):
        return LyteProfileV6(join(data_dir, filename), calibration={'Sensor1': [1, 0]})

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 9539)
    ])
    def test_start_property(self, profile, filename, expected):
        start = profile.start.index
        assert start == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 27278)
    ])
    def test_stop_property(self, profile, filename, expected):
        stop = profile.stop.index
        assert profile.stop.index == expected

    @pytest.mark.parametrize('filename, expected_idx', [
        ('kaslo.csv', 1832)
    ])
    def test_nir_surface_property(self, profile, filename, expected_idx):
        nir_surface = profile.surface.nir
        assert profile.surface.nir.index ==expected_idx

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 118.5)
    ])
    def test_distance_through_snow(self, profile, expected):
        delta = profile.distance_through_snow
        assert pytest.approx(delta, abs=0.5) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 123)
    ])
    def test_distance_traveled(self, profile, expected):
        delta = profile.distance_traveled
        assert pytest.approx(delta, abs=0.5) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 112)
    ])
    def test_avg_velocity(self, profile, expected):
        delta = profile.avg_velocity
        assert pytest.approx(delta, abs=0.5) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 1.1)
    ])
    def test_moving_time(self, profile, expected):
        delta = profile.moving_time
        assert pytest.approx(delta, abs=0.01) == expected

    @pytest.mark.parametrize('filename, expected_points, mean_force', [
        ('kaslo.csv', 17269, 3548.3813)
    ])
    def test_force_profile(self, profile, filename, expected_points, mean_force):
        assert len(profile.force) == expected_points
        assert pytest.approx(profile.force['force'].mean(), abs=1e-4) == mean_force

    @pytest.mark.parametrize('filename, expected_points, mean_force', [
        ('kaslo.csv', 12979, 3027.4536)
    ])
    def test_nir_profile(self, profile, filename, expected_points, mean_force):
        print(profile.nir)
        assert len(profile.nir) == expected_points
        assert pytest.approx(profile.nir['nir'].mean(), abs=1e-4) == mean_force


    @pytest.mark.parametrize('columns, expected', [
        # Test old naming of accelerometer
        (['acceleration'], 'acceleration'),
        # Test newer naming
        (['Y-Axis', 'X-Axis', 'Z-Axis'], 'Y-Axis'),
        # Test no accelerometer
        (['Sensor1', 'Sensor2'], None),
    ])
    def test_get_accelerometer_column(self, columns, expected):
        """ Test the retrieval of the accelerometer column"""
        result = LyteProfileV6.get_accelerometer_column(columns)
        assert expected == result