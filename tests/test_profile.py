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
        ('kaslo.csv', 123.5)
    ])
    def test_distance_through_snow(self, profile, expected):
        delta = profile.distance_through_snow
        assert pytest.approx(delta, abs=1) == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 123)
    ])
    def test_distance_traveled(self, profile, expected):
        delta = profile.distance_traveled
        assert pytest.approx(delta, abs=1) == expected

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
        ('kaslo.csv', 14799, 2863)
    ])
    def test_nir_profile(self, profile, filename, expected_points, mean_force):
        assert pytest.approx(len(profile.nir), len(profile.raw)*0.005) == expected_points
        assert pytest.approx(profile.nir['nir'].mean(), abs=3) == mean_force


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

    @pytest.mark.parametrize('filename', ['kaslo.csv'])
    def test_barometer(self, profile):
        assert pytest.approx(profile.barometer.distance_traveled, 1e-2) == 116.00

    @pytest.mark.parametrize('filename', ['kaslo.csv'])
    def test_accelerometer(self, profile):
        assert pytest.approx(profile.accelerometer.distance_traveled, 1e-2) == 124.54

    @pytest.mark.parametrize('filename', ['kaslo.csv'])
    def test_depth_profiles_are_independent(self, profile):
        """Issue in merged depth shows up if the data is not a copy"""
        # Invoke the fusing of depths where leaking data would occur
        profile.depth
        delta = profile.accelerometer.depth.iloc[profile.error.index] - profile.barometer.depth.iloc[profile.error.index]
        # Profiles are different enough that it should be more than a 0.5cm
        assert delta > 0.5


def test_old_profile(data_dir):
    """
    Test profile is able to compute surface and stop from older
    no acceleration data
    """
    f = 'old_probe.csv'
    p = join(data_dir, f)
    profile = LyteProfileV6(p)
    assert profile.stop.index == 29685
    assert profile.surface.nir.index == 7970