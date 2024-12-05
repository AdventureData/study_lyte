import numpy as np
import pytest
import pandas as pd
from study_lyte.cropping import crop_to_motion, crop_to_snow


@pytest.mark.parametrize('fname, start_kwargs, stop_kwargs, expected_time_delta', [
    ('bogus.csv', {}, {}, 0.98),
])
def test_crop_to_motion(raw_df, fname, start_kwargs, stop_kwargs, expected_time_delta):
    """
    Test that the dataframe is cropped correctly according to motion
    then compare with the timing
    """
    df = crop_to_motion(raw_df, start_kwargs=start_kwargs, stop_kwargs=stop_kwargs)
    delta_t = df.index.max() - df.index.min()

    assert pytest.approx(delta_t, abs=0.02) == expected_time_delta