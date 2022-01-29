from study_lyte.detect import get_signal_event
import pytest
import numpy as np
import pandas as pd


@pytest.mark.parametrize("data, threshold, direction, expected", [
    (np.array([0, 1, 0]), 0.5, 'forward', 1),
    (pd.Series([1, 0, 0]), 0.5, 'forward', 0),
    (np.array([0, 0, 0.1]), 0.01, 'backwards', 2),
    (np.array([20, 100, 100, 40]), 50, 'backwards', 2),
])
def test_get_signal_event(data, threshold, direction, expected):
    """
    Test the signal event is capable of return the correct index
    regardless of threshold, direction, and series or numpy array
    """
    idx = get_signal_event(data, threshold=threshold, search_direction=direction)
    assert idx == expected

    