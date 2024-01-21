import pytest
from os.path import join
from study_lyte.profile import LyteProfileV6, Sensor
from operator import attrgetter


class TestLyteProfile:

    @pytest.fixture()
    def profile(self, data_dir, filename, depth_method):
        return LyteProfileV6(join(data_dir, filename), calibration={'Sensor1': [1, 0]}, depth_method=depth_method)

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 9539)
    ])
    def test_start_property(self, profile, filename, expected):
        start = profile.start.index
        # Loose tolerances more about testing functionality
        assert  pytest.approx(start, abs=len(profile.raw)*0.05) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 27278)
    ])
    def test_stop_property(self, profile, filename, depth_method, expected):
        stop = profile.stop.index
        assert pytest.approx(stop, abs= len(profile.raw)*0.005) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 12479)
    ])
    def test_nir_surface_property(self, profile, filename, depth_method, expected):
        nir_surface = profile.surface.nir.index
        assert  pytest.approx(nir_surface, abs=len(profile.raw)*0.01) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 118)
    ])
    def test_distance_through_snow(self, profile, expected):
        delta = profile.distance_through_snow
        assert pytest.approx(delta, abs=2.5) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Test our default method
        ('kaslo.csv', 'fused', 119),
        # Test our extra methods
        ('kaslo.csv', 'accelerometer', 125),
        ('kaslo.csv', 'barometer', 116.00)
    ])
    def test_distance_traveled(self, profile, expected):
        delta = profile.distance_traveled
        assert pytest.approx(delta, abs=2.5) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 108.6)
    ])
    def test_avg_velocity(self, profile, expected):
        delta = profile.avg_velocity
        assert pytest.approx(delta, abs=5) == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 1.1)
    ])
    def test_moving_time(self, profile, expected):
        delta = profile.moving_time
        assert pytest.approx(delta, abs=0.05) == expected

    @pytest.mark.parametrize('filename, depth_method, expected_points, mean_force', [
        ('kaslo.csv', 'fused', 16377, 3500)
    ])
    def test_force_profile(self, profile, filename, depth_method, expected_points, mean_force):
        assert pytest.approx(len(profile.force), len(profile.raw)*0.05) == expected_points
        assert pytest.approx(profile.force['force'].mean(), abs=150) == mean_force

    @pytest.mark.parametrize('filename, depth_method, expected_points, mean_pressure', [
        ('kaslo.csv', 'fused', 16377, 179)
    ])
    def test_pressure_profile(self, profile, filename, depth_method, expected_points, mean_pressure):
        assert pytest.approx(len(profile.pressure), len(profile.raw)*0.05) == expected_points
        assert pytest.approx(profile.pressure['pressure'].mean(), abs=10) == mean_pressure


    @pytest.mark.parametrize('filename, depth_method, expected_points, mean_nir', [
        ('kaslo.csv', 'fused', 14799, 2863)
    ])
    def test_nir_profile(self, profile, filename, depth_method, expected_points, mean_nir):
        assert pytest.approx(len(profile.nir), len(profile.raw)*0.05) == expected_points
        assert pytest.approx(profile.nir['nir'].mean(), abs=50) == mean_nir


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

    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'fused')
    ])
    def test_repr(self, profile):
        profile_str = f"LyteProfile (Recorded {len(profile.raw):,} points, {profile.datetime.isoformat()})"
        assert str(profile) == profile_str

    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'fused')
    ])
    def test_barometer(self, profile):
        result = pytest.approx(profile.barometer.distance_traveled, 1e-2)
        assert result == 116.00


    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'fused')
    ])
    def test_accelerometer(self, profile):
        assert pytest.approx(profile.accelerometer.distance_traveled, abs=2.5) == 124.54
    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'fused')
    ])
    def test_depth_profiles_are_independent(self, profile):
        """Issue in merged depth shows up if the data is not a copy"""
        # Invoke the fusing of depths where leaking data would occur
        profile.depth
        delta = profile.accelerometer.depth.iloc[profile.error.index] - profile.barometer.depth.iloc[profile.error.index]
        # Profiles are different enough that it should be more than a 0.5cm
        assert abs(delta) > 0.5

    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'barometer'),
        ('kaslo.csv', 'accelerometer'),
        ('kaslo.csv', 'fused')
    ])
    def test_recursion_exceedance_on_depth(self, profile, filename, depth_method):
        """This confirms we don't have a sneaky recursion error around selecting depth methods"""
        # Invoking force while picking depth method once caused issues.
        try:
            profile.force
        except Exception:
            pytest.fail("Unable to invoke profile.force, likely an recursion issue...")

    @pytest.mark.parametrize('filename, depth_method, total_depth', [
        # Is filtered
        ('egrip.csv', 'fused', 199),
        # Not filtered
        ('kaslo.csv','fused', 116),
    ])
    def test_barometer_is_filtered(self, profile, filename, depth_method, total_depth):
        assert pytest.approx(profile.barometer.distance_traveled, abs=1) == total_depth

    @pytest.mark.parametrize('filename, depth_method', [
        # Chooses start over force start
        ('kaslo.csv','fused'),
        # Chooses start over force start
        ('toolik.csv', 'fused'),
    ])
    def test_force_start_modification(self, profile, filename, depth_method):
        """Test the profile will choose an earlier force surface should it exist prior to the NIR"""
        # Ensure that the sensors detection is always in order.
        assert profile.start.index <= profile.surface.force.index
        assert profile.surface.force.index <= profile.surface.nir.index

    def test_isolated_reading_metadata(self, data_dir):
        """ Test the metadata can be read independently without parsing the whole file"""
        profile = LyteProfileV6(join(data_dir, 'toolik.csv'))
        metadata = profile.metadata
        assert type(metadata) == dict
        assert profile._raw is None

class TestLegacyProfile:
    @pytest.fixture()
    def profile(self, data_dir):
        f = 'old_probe.csv'
        p = join(data_dir, f)
        profile = LyteProfileV6(p)
        return profile

    def test_stop_wo_accel(self, profile):
        """
        Test profile is able to compute surface and stop from older
        no acceleration data
        """
        assert pytest.approx(profile.stop.index, int(0.01*len(profile.depth))) == 29685

    def test_surface(self, profile):
        assert pytest.approx(profile.surface.nir.index, int(0.01*len(profile.depth))) == 7970

@pytest.mark.parametrize('fname, attribute, expected_value', [
    ('pilots_error.csv', 'surface.force.index', 5758)
])
def test_force_start_alternate(lyte_profile, fname, attribute, expected_value):
    result =  attrgetter(attribute)(lyte_profile)
    assert pytest.approx(result, int(0.01 * len(lyte_profile.raw))) == expected_value
