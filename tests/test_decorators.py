from study_lyte.decorators import directional, time_series
import pytest
import pandas as pd


@pytest.mark.parametrize('direction, raises_error', [
    ('forward', False),
    ('backward', False),
    ('backwards', True),

])
def test_directional(direction, raises_error):

    @directional(check='search_direction')
    def search(search_direction='forwards'):
        return 10

    if raises_error:
        with pytest.raises(ValueError):
            search(search_direction=direction)
    else:
        assert search(search_direction=direction)


@pytest.mark.parametrize('df, raises_error', [
    # Test setting time as the index
    (pd.DataFrame({'time': [1], 'data': [1]}), False),
    # Test observing time is the index
    (pd.DataFrame({'time': [1], 'data': [1]}).set_index('time'), False),
    # Test no time index error
    (pd.DataFrame({'data': [1]}), True),

])
def test_time_series_decorator(df, raises_error):
    @time_series
    def get_df(data):
        return data

    if raises_error:
        with pytest.raises(ValueError):
            get_df(df)
    else:
        assert get_df(df).index.name == 'time'