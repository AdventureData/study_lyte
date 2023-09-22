import numpy as np
from string import ascii_uppercase

class LinearRegression:
    """
    Class for managing simple linear regressions
    """
    def __init__(self, coefficients=None, predicted_name=None, coefficient_names=None):
        """
        Args:
            coefficients: If a regression has already be formed then instantiate from there.
            predicted_name: Name to use in various strings for outputting name.
            coefficient_names: Names of the coefficients to use in various strings.

        """
        self._coefficients = coefficients
        self._name = predicted_name
        self._coefficient_names = coefficient_names
        # String holder for a plain text equation and one renderable for latex
        self._equation = None
        self._rendered_equation = None


    @property
    def coefficients(self):
        """Resulting coefficients from the regression"""
        if self._coefficients is None:
            self._coefficients = []
        return self._coefficients

    @property
    def coefficient_names(self):
        """name to use in strings outputting information"""
        if self._coefficient_names is None:
            # Use made up variables.
            if self.coefficients:
                self._coefficient_names = [f"{ascii_uppercase[-1*(i+1)]}" for i,c in enumerate(self.coefficients[0:-1])]

        if len(self._coefficient_names) != len(self.coefficients):
            self._coefficient_names.append('')

        return self._coefficient_names

    @property
    def name(self):
        """name to use in strings outputting information"""
        if self._name is None:
            self._name = 'data'
        return self._name

    @property
    def equation(self):
        """Useful string representation of resulting equation"""
        if self._equation is None:
            self._equation = self.get_equation_string()
        return self._equation

    @property
    def rendered_equation(self):
        """Useful string representation"""
        if self._rendered_equation is None:
            self._rendered_equation = self.get_equation_string(rendered=True)
        return self._rendered_equation

    def regress(self, input_df, output_series):
        """
        Take a pandas dataframe and form a regression against the output series
        data
        Args:
            input_df: Pandas Dataframe containing inputs to the regression. Column names are used in equation.
            if columns names are written in latex, they can be rendered in rendered equation for matplotlib
            output_series: Data to regress against
        """
        columns = list(input_df.columns)
        # Set string columns names anytime we regress. Clear our equation string
        self._coefficient_names = columns
        self._equation = None
        self._rendered_equation = None

        a_matrix = np.vstack([input_df[c] for c in columns] + [np.ones(len(input_df.index))]).T
        self._coefficients = list(np.linalg.lstsq(a_matrix, output_series, rcond=None)[0])

    def predict(self, input_df):
        """
        Use the regression to predict data
        Args:
            input_df: Pandas Dataframe containing inputs for the regression. Columns are assumed in same order
                as regression coefficients
        Returns:
            result: Resulting data predicted by the regression
        """
        result = 0
        for coefficient, column in zip(self.coefficients[0:-1], input_df.columns):
            result += coefficient * input_df[column]
        result += self.coefficients[-1] * np.ones_like(input_df.index)
        return result

    @staticmethod
    def quality(predicted, measured):
        """
        Perform some quality metrics against some predicted and measured data
        Args:
            predicted: Series of predicted data
            measured: Series of measured data (same length as predicted)
        Returns:
            dictionary of varius performance metrics
        """
        m_mean = measured.mean()
        p_mean = predicted.mean()
        diff = p_mean - m_mean

        # Perform point by point metrics
        series_difference = predicted - measured
        p_difference = series_difference / measured

        # Comparisons against absolute values
        abs_diff = abs(series_difference)
        p_abs_diff = abs_diff / measured

        return {"mean difference":{'value':diff, "percent": diff / m_mean},
                # point by point differences
                'mean point error': {'value':series_difference.mean(), 'percent':p_difference.mean()},
                'max point error': {'value':series_difference.max(),'percent': p_difference.max()},
                'min point error': {'value':series_difference.min(),'percent':p_difference.min()},
                # Absolute value
                'mean absolute point error': {'value': abs_diff.mean(), 'percent': p_abs_diff.mean()},
                'max absolute point error': {'value': abs_diff.max(), 'percent': p_abs_diff.max()},
                'min absolut point error': {'value': abs_diff.min(), 'percent': p_abs_diff.min()},
                }

    @staticmethod
    def quality_report(results):
        """
        Args:
            results: Dictionary from quality function
        Returns:
            string of quality info
        """
        string_report = ''
        for k,v in results.items():
            string_report += f"{k.title()} = {v['value']:0.2f} ({v['percent']:0.2%})"
        return string_report

    def get_equation_string(self, rendered=False):
        """
        Generate the string representation for the equation.
        Args:
            rendered: Keep any render indicators like $ \ etc to get greek repr and various subscripts to be rendered in things
                like matplotlib. Otherwise remove them.
        Returns:
            equation: string equation
        """
        # No coefficients available, then there is no known fit
        if not self.coefficients:
            result = f'{self.name.title()} (Pre-fit)'
        else:
            result = f"{self.name} = "
            i = 0
            for c, v in zip(self.coefficients, self.coefficient_names):
                c_str = f"{abs(c):0.3f}"

                if v:
                    if not rendered:
                        # Remove any render indicators e.g. r"$\rho_E$
                        variable = v.replace('$', '').replace("\\", '')
                    else:
                        variable = v

                    term = rf"{c_str}*{variable}"

                else:
                    term = c_str

                if i == 0:
                    joiner = ''
                else:
                    if c < 0:
                        joiner = ' - '
                    else:
                        joiner = ' + '

                result += f'{joiner}{term}'
                i += 1

        return result

    def __repr__(self):
        return f'Linear Regression: {self.equation}'