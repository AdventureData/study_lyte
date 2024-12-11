import pytest
from os.path import join
from pathlib import Path
from study_lyte.calibrations import Calibrations
from study_lyte.profile import ProcessedProfileV6, LyteProfileV6, Sensor
from operator import attrgetter
from shapely.geometry import Point

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
        ('kaslo.csv', 'fused', 11641)
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
        ('kaslo.csv', 'fused', 14799, 2638)
    ])
    def test_nir_profile(self, profile, filename, depth_method, expected_points, mean_nir):
        # Assert the len of the nir profile within 5%
        assert pytest.approx(len(profile.nir), len(profile.raw) * 0.05) == expected_points
        # Use the median as an approximation for the processing
        assert pytest.approx(profile.nir['nir'].median(), abs=50) == mean_nir
        
    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Serious angle
        ('angled_measurement.csv', 'fused', 34),
        # Nearly vertical
        ('toolik.csv', 'fused', 2),
        ('kaslo.csv', 'fused', Sensor.UNAVAILABLE)
    ])
    def test_angle_attribute(self, profile, filename, depth_method, expected):
        assert pytest.approx(profile.angle, abs=2) == expected

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

    @pytest.mark.parametrize('filename, depth_method, expected', [
        ('kaslo.csv', 'fused', 116),
        ('angled_measurement.csv', 'fused', 44),
    ])
    def test_barometer(self, profile, expected):
        result = pytest.approx(profile.barometer.distance_traveled, 1e-2)
        assert result == expected


    @pytest.mark.parametrize('filename, depth_method', [
        ('kaslo.csv', 'fused'),
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
        # ('kaslo.csv','fused', 116),
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

    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Parseable point
        ('ground_touch_and_go.csv','fused', Point(-115.693, 43.961)),
        # no point available
        ('egrip.csv', 'fused', Sensor.UNAVAILABLE),
    ])
    def test_point(self, profile, filename, depth_method, expected):
        """Test we are parsing the point info"""
        assert profile.point == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Test report card doesn't error out when missing key features
        ("open_air.csv", 'fused', 0),
    ])
    def test_report_card(self, profile, filename, depth_method, expected):
        """Test we are parsing the point info"""
        result = profile.report_card()
        assert True

    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Test assignment of the serial number from file
        ("open_air.csv", 'fused', "252813070A020004"),
        # Test serial number not found
        ("mores_20230119.csv", 'fused', "UNKNOWN"),
    ])
    def test_serial_number(self, profile,filename, depth_method, expected):
        assert profile.serial_number == expected

    @pytest.mark.parametrize('filename, depth_method, expected', [
        # Test assignment of the serial number from file
        ("open_air.csv", 'fused', -10),
        # Test serial number not found
        ("mores_20230119.csv", 'fused', -1),
    ])
    def test_set_calibration(self, data_dir, profile,filename, depth_method, expected):
        p = Path(join(data_dir,'calibrations.json'))
        calibrations = Calibrations(p)
        profile.set_calibration(calibrations)
        assert profile.calibration['Sensor1'][2] == expected


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


@pytest.mark.parametrize("fname, expected", [("open_air.csv", 0)])
def test_surface_indexer_error(lyte_profile, fname, expected):
    """
    Most people take a measure in the air to test which fails to find a surface.
    Test to make sure this is handled.
    """
    assert lyte_profile.surface.nir.index == expected
    assert not lyte_profile.nir.empty


@pytest.mark.skip('Incomplete work')
def test_app(data_dir):
    """Functionality test"""
    fname = data_dir + '/ls_app.csv'
    profile = ProcessedProfileV6(fname)
    print(profile)
    assert False # TODO: Add more detailed checking