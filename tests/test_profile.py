import pytest
from os.path import join
from study_lyte.profile import LyteProfileV6


class TestLyteProfile:

    @pytest.fixture()
    def profile(self, data_dir, filename):
        return LyteProfileV6(join(data_dir, filename))

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 100)
    ])
    def test_start_property(self, profile, filename, expected):
        assert profile.start.index == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 100)
    ])
    def test_stop_property(self, profile, filename, expected):
        assert profile.stop.index == expected

    @pytest.mark.parametrize('filename, expected', [
        ('kaslo.csv', 100)
    ])
    def test_surface_property(self, profile, filename, expected):
        from study_lyte.plotting import plot_ts
        #plot_ts(profile.cropped['Sensor1'], events=[(e.name, e.index) for e in profile.events])
        print(profile)
        assert profile.surface.index == expected