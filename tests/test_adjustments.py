from study_lyte.adjustments import get_directional_mean, get_neutral_bias_at_border
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
