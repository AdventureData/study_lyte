import pytest

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