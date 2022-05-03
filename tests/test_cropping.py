import numpy as np
import pytest
import pandas as pd
from study_lyte.cropping import crop_to_motion, crop_to_snow


@pytest.mark.parametrize('fname, start_kwargs, stop_kwargs, expected_time_delta', [
    ('raw_depth_data_short.csv', {}, {}, 0.0676897922180939),
])
def test_crop_to_motion(raw_df, fname, start_kwargs, stop_kwargs, expected_time_delta):
    """
    Test that the dataframe is cropped correctly according to motion
    then compare with the timing
    """
    df = crop_to_motion(raw_df, start_kwargs=start_kwargs, stop_kwargs=stop_kwargs)
    assert (df.index.max() - df.index.min()) == expected_time_delta


@pytest.mark.parametrize('active, ambient, expected_surface_index', [
    ([2, 2, 1, 1], [1, 1, 2, 2], 3),
])
def test_crop_to_snow(active, ambient, expected_surface_index):
    """
    Test that the dataframe is cropped correctly according to motion
    then compare with the timing
    """
    data = {'time': np.linspace(0, 1, len(active)),
            'active': np.array(active),
            'ambient': np.array(ambient)}
    df = pd.DataFrame(data)
    cropped = crop_to_snow(df, active_col='active', ambient_col='ambient', fractional_basis=0.25)
    assert (df.index.max() - df.index.min()) == expected_surface_index
