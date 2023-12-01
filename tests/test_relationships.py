import pandas as pd
import  numpy as np

from study_lyte.relationships import LinearRegression
import pytest

class TestLinearRegression:
    @pytest.fixture(scope='class')
    def input_data(self):
        df = pd.DataFrame({r'$\lambda$': [1, 2, 3]})
        return df

    @pytest.fixture(scope='class')
    def measured(self):
        series = pd.Series([21, 41, 61])
        return series

    @pytest.fixture(scope='class')
    def relationship(self, input_data, measured):
        rel = LinearRegression(predicted_name='alpha')
        rel.regress(input_data, measured)
        return rel

    @pytest.fixture(scope='class')
    def relationship_predefined(self, input_data, measured):
        """
        Data relationship that came with a fit already determined, no names provided
        """
        rel = LinearRegression(coefficients=[10, 11, 1])
        return rel

    def test_coefficients(self, relationship):
        """
        Test the coefficients are available and correct
        """
        np.testing.assert_almost_equal(relationship.coefficients, [20, 1], decimal=5)

    def test_prediction(self, relationship):
        """
        Test the prediction function is correct
        """
        result = relationship.predict(pd.DataFrame({'Force':[4, 5, 6]}))
        np.testing.assert_almost_equal(result, [81, 101, 121], decimal=5)

    @pytest.mark.parametrize('key, second_key, expected', [
        # Confirm a few stats are in here
        ('mean difference', 'value', -3.33333),
        ('max absolute point error', 'value', 20),
        ('mean point error', 'percent', -0.00387),
    ])
    def test_quality(self, relationship, measured, key, second_key, expected):
        """
        Test the prediction function is correct
        """
        predicted = pd.Series([31, 21, 61])
        result = relationship.quality(predicted, measured)
        np.testing.assert_almost_equal(result[key][second_key], expected, decimal=5)

    def test_representation_after_regression(self, relationship):
        """
        Test the string representation works
        """
        string_eq = relationship.equation
        assert string_eq == 'alpha = 20.000*lambda + 1.000'

    def test_renderable_equation(self, relationship):
        """
        Test the string representation works
        """
        string_eq = relationship.rendered_equation
        assert '$' in string_eq
        assert '$' not in relationship.equation

    def test_representation_before_regression(self):
        """
        Test the string representation works
        """
        rel = LinearRegression()
        string_eq = str(rel)
        assert string_eq == 'Linear Regression (N = Unknown): Data (Pre-fit)'

    def test_equation_predefined(self, relationship_predefined):
        """
        Test the string representation works
        """
        string_eq = relationship_predefined.equation
        assert string_eq == 'data = 10.000*Z + 11.000*Y + 1.000'
