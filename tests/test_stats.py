from study_lyte.stats import required_sample_for_margin, margin_of_error
import pytest


@pytest.mark.parametrize('n, std, confidence, expected', [
    (5, 8.9, 0.95, 7.80)
])
def test_margin_of_error(n, std, confidence, expected):
    result = margin_of_error(n, std, confidence=confidence)
    assert pytest.approx(result, abs=1e-2) == expected


@pytest.mark.parametrize('desired_margin, std, confidence, expected', [
    (5, 8.9, 0.95, 12.17)
])
def test_required_sample_for_margin(desired_margin, std, confidence, expected):
    n = required_sample_for_margin(desired_margin, std, confidence=confidence)
    assert pytest.approx(n, abs=1e-2) == expected