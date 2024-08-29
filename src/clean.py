from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class RareCategoryReplacer(BaseEstimator, TransformerMixin):
    """
    A transformer class for replacing rare categories in specified columns of a DataFrame.

    Parameters:
    ----------
    columns : list
        List of column names to apply the rare category replacement.
    proportion_threshold : float, optional (default=0.02)
        Threshold below which a category is considered rare.
    replacement_value : str, optional (default="Others")
        Value to replace rare categories with.

    Attributes:
    ----------
    rare_categories_ : dict
        Dictionary containing the rare categories for each specified column.
    important_categories_ : dict
        Dictionary containing the important categories for each specified column.

    Methods:
    -------
    fit(X, y=None)
        Fit the transformer to the data by calculating the rare categories and important categories for each specified column.
    transform(X)
        Transform the data by replacing rare categories with the replacement value.
    fit_transform(X, y=None)
        Fit the transformer to the data and transform it in a single step.
    """
    def __init__(self, columns, proportion_threshold=0.02, replacement_value="Others"):
        self.columns = columns
        self.proportion_threshold = proportion_threshold
        self.replacement_value = replacement_value
        self.rare_categories_ = {}
        self.important_categories_ = {}

    def fit(self, X, y=None):
        # Calculate the percentage of each category for each specified column
        for column in self.columns:
            category_percentages = X[column].value_counts(normalize=True)
            self.rare_categories_[column] = category_percentages[
                category_percentages < self.proportion_threshold
            ].index.tolist()
            self.important_categories_[column] = category_percentages[
                category_percentages >= self.proportion_threshold
            ].index.tolist()

        return self

    def transform(self, X):
        X = X.copy()  # Create a copy of the DataFrame to avoid modifying the original data

        for column in self.columns:
            # Replace rare categories with the replacement value
            X[column] = np.where(
                X[column].isin(self.rare_categories_[column]), 
                self.replacement_value, 
                X[column]
            )

            # Replace any new categories not in the important categories with the replacement value
            allowed_categories = self.important_categories_[column] + [self.replacement_value]
            X[column] = np.where(
                ~X[column].isin(allowed_categories), 
                self.replacement_value, 
                X[column]
            )

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# Example usage:
# columns_to_replace = ['cap-shape', 'cap-color', 'gill-color']
# replacer = RareCategoryReplacer(columns=columns_to_replace, proportion_threshold=0.02)
# df_train = replacer.fit_transform(df_train)  # Fit and transform on training data
# df_test = replacer.transform(df_test)        # Transform test data using the same categories
