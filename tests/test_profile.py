import pytest
from os.path import join
from study_lyte.profile import LyteProfileV6, Sensor


class TestLyteProfile:

    @pytest.fixture()
    def profile(self, data_dir, filename):
        return LyteProfileV6(join(data_dir, filename), calibration={'Sensor1': [1, 0]})

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 9539)
    ])
    def test_start_property(self, profile, filename, expected):
        start = profile.start.index
        assert  pytest.approx(start, abs= len(profile.raw)*0.005) == expected


    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 27278)
    ])
    def test_stop_property(self, profile, filename, expected):
        stop = profile.stop.index
        assert pytest.approx(stop, abs= len(profile.raw)*0.005) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 12479)
    ])
    def test_nir_surface_property(self, profile, filename, expected):
        nir_surface = profile.surface.nir.index
        assert  pytest.approx(nir_surface, abs=len(profile.raw)*0.005) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 122.5)
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
        ('kaslo.csv', 16377, 3521)
    ])
    def test_force_profile(self, profile, filename, expected_points, mean_force):
        assert pytest.approx(len(profile.force), len(profile.raw)*0.005) == expected_points
        assert pytest.approx(profile.force['force'].mean(), abs=10) == mean_force

    @pytest.mark.parametrize('filename, expected_points, mean_force', [
        ('kaslo.csv', 14799, 2860.5104)
    ])
    def test_nir_profile(self, profile, filename, expected_points, mean_force):
        assert pytest.approx(len(profile.nir), len(profile.raw)*0.005) == expected_points
        assert pytest.approx(profile.nir['nir'].mean(), abs=1) == mean_force


    @pytest.mark.parametrize('columns, expected', [
        # Test old naming of accelerometer
        (['acceleration'], 'acceleration'),
        # Test newer naming
        (['Y-Axis', 'X-Axis', 'Z-Axis'], 'Y-Axis'),
        # Test no accelerometer
        (['Sensor1', 'Sensor2'], Sensor.UNAVAILABLE),
    ])
    def test_get_motion_name(self, columns, expected):
        """ Test the retrieval of the accelerometer column"""
        result = LyteProfileV6.get_motion_name(columns)
        assert result == expected

    @pytest.mark.parametrize('columns, expected', [
        # Test old naming of accelerometer
        (['acceleration'], ['acceleration']),
        # Test newer naming
        (['Y-Axis', 'X-Axis', 'Z-Axis'], ['X-Axis', 'Y-Axis', 'Z-Axis']),
        # Test no accelerometer
        (['Sensor1', 'Sensor2'], Sensor.UNAVAILABLE),
    ])
    def test_get_motion_name(self, columns, expected):
        """ Test the retrieval of the accelerometer column"""
        result = LyteProfileV6.get_acceleration_columns(columns)
        assert result == expected

    @pytest.mark.parametrize('filename', ['kaslo.csv'])
    def test_repr(self, profile):
        profile_str = f"LyteProfile (Recorded {len(profile.raw):,} points, {profile.datetime.isoformat()})"
        assert str(profile) == profile_str
