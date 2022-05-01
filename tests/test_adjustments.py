from study_lyte.adjustments import (get_directional_mean, get_neutral_bias_at_border, get_normalized_at_border, \
                                    merge_time_series)
import pytest
import pandas as pd
import numpy as np


@pytest.mark.parametrize('data, fractional_basis, direction, expected', [
    # Test the directionality
    ([1, 1, 2, 2], 0.5, 'forward', 1),
    ([1, 1, 2, 2], 0.5, 'backward', 2),
    #  fractional basis
    ([1, 3, 4, 6], 0.25, 'forward', 1),
    ([1, 3, 5, 6], 0.75, 'forward', 3)
])
def test_directional_mean(data, fractional_basis, direction, expected):
    """
    Test the directional mean function
    """
    df = pd.DataFrame({'data': np.array(data)})
    value = get_directional_mean(df['data'], fractional_basis=fractional_basis, direction=direction)
    assert value == expected


@pytest.mark.parametrize('data, fractional_basis, direction, zero_bias_idx', [
    # Test the directionality
    ([1, 1, 2, 2], 0.5, 'forward', 0),
    ([1, 1, 2, 2], 0.5, 'backward', -1),
])
def test_get_neutral_bias_at_border(data, fractional_basis, direction, zero_bias_idx):
    """
    Test getting neutral bias at the borders of the data, use the zero_bias_idx to assert
    where the bias of the data should be zero
    """
    df = pd.DataFrame({'data': np.array(data)})
    result = get_neutral_bias_at_border(df['data'], fractional_basis=fractional_basis, direction=direction)
    assert result.iloc[zero_bias_idx] == 0

@pytest.mark.parametrize('data, fractional_basis, direction, ideal_norm_index', [
    # Test the directionality
    ([1, 1, 2, 2], 0.5, 'forward', 0),
    ([1, 1, 2, 2], 0.5, 'backward', -1),
])
def test_get_normalized_at_border(data, fractional_basis, direction, ideal_norm_index):
    """
    Test getting a dataset normalized by its border where the border values of the data should be one
    """
    df = pd.DataFrame({'data': np.array(data)})
    result = get_normalized_at_border(df['data'], fractional_basis=fractional_basis, direction=direction)
    assert result.iloc[ideal_norm_index] == 1


@pytest.mark.parametrize('data_list, expected', [
    # Typical use, low sample to high res
    ([np.linspace(1, 4, 4), np.linspace(1, 4, 2)], 2 * [np.linspace(1, 4, 4)]),
    # No data
    ([], []),
])
def test_merge_time_series(data_list, expected):
    # build a convenient list of dataframes
    other_dfs = []
    expected_dict = {}
    for i, data in enumerate(data_list):
        name = f'data_{i}'
        # Let the number of values determine the sampling of time, always across 1 second
        ts = np.linspace(0, 1, len(data))
        df = pd.DataFrame({'time': ts, name: data})
        other_dfs.append(df)

        # Build the expected dictionary
        expected_dict[name] = expected[i]

    result = merge_time_series(other_dfs)

    # Build the expected using the expected input
    expected_df = pd.DataFrame(expected_dict)
    exp_cols = expected_df.columns

    pd.testing.assert_frame_equal(result[exp_cols], expected_df)