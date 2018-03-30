import numpy as np
np.random.seed(42)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


UNSHIFTED_COLS = ['team', 'oppo_team', 'last_finals_reached', 'venue', 'home_team',
                  'year', 'round_number', 'win_odds', 'point_spread']


class ColumnFilter(BaseEstimator, TransformerMixin):
    """
    Returns a Numpy array with columns based on column names matching a regex string.
    Since this relies on columns having names, which means the input must be a
    DataFrame, this must be the first transformer in a pipeline (or have a custom
    transformer before it that returns a DataFrame).

    Parameters
    ----------
        include : {string}
            String representation of a regular expression for matching column names.
            Matched columns will be passed to the next transformer or model.
        exclude : {string}
            String representation of a regular expression for matching column names.
            Matched columns will be dropped from the DataFrame.
    """

    def __init__(self, include=None, exclude=None, col_order=None):
        self.include = include or '.*' if exclude is None else include
        self.exclude = exclude
        self.col_order = col_order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Parameters
        ----------
            X : {pandas DataFrame}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            {Numpy array}, shape = [n_observations, n_filtered_features]
                Samples with matching columns only
        """

        # If both params are used, include takes precedence
        if self.include is not None:
            X_filtered = X.filter(regex=self.include)
        elif self.exclude is not None:
            drop_cols = X.filter(regex=self.exclude).columns.values
            X_filtered = X.drop(drop_cols, axis=1)
        else:
            return X.values

        # Sometimes its important to have category columns in front for segmentation
        # purposes
        if self.col_order is not None:
            filtered_cols = [col for col in X_filtered.columns.values if col not in self.col_order]
            ordered_cols = self.col_order + filtered_cols
            X_filtered = X_filtered[ordered_cols]

        return X_filtered.values


class TimeStepReshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes the sample data matrix from [n_observations, n_features] to
    [n_observations, n_steps, n_features] (via [n_segments, n_observations / n_segments, n_features])
    and returns the 3D matrix for use in a RNN model.

    Parameters
    ----------
        n_steps : {int}
            Number of steps back in time added to each observation.
        segment_col : {int}
            Index of the column with the segment values.
        are_labels : {boolean}
            Whether data are features or labels.

    """

    def __init__(self, n_steps=None, are_labels=False, segment_col=0, rnn=True):
        if n_steps is None:
            raise(Exception('n_steps is None. Be sure to explicitly set n_steps ' +
                            'to avoid data shape errors.'))

        self.n_steps = n_steps
        self.are_labels = are_labels
        self.segment_col = segment_col
        self.rnn = rnn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            {array-like}, shape = [n_observations, n_steps, n_features]
                Samples with matching columns only
        """

        X_values = X.values if X.__class__.__name__ == 'DataFrame' else X

        # [n_segments, n_observations / n_segments, n_features]
        X_segmented = self.__segment_arrays(X_values)

        time_step_matrix = np.array(
            # Shift by time steps: [n_steps, n_segments, n_observations / n_segments, n_features]
            [X_segmented[:, self.n_steps - step_n:-step_n or None, :] for step_n in range(self.n_steps)]
        ).reshape(
            # Reshape to eliminate segmentation: [n_steps, n_observations, n_features]
            self.n_steps, -1, X_segmented.shape[-1]
        ).transpose(
            # Transpose into shape for RNN: [n_observations, n_steps, n_features]
            1, 0, 2
        )

        # Return 3D data matrix for RNN
        if self.rnn:
            return time_step_matrix

        # If transforming for ML model, return 2D data matrix with time-step columns
        # (shape: [n_observations, n_features * n_steps])
        return time_step_matrix.reshape(time_step_matrix.shape[0], -1)

    def __segment_arrays(self, X):
        # If input is labels instead of features, ignore all but last column,
        # because others are only there for segmenting purposes
        slice_start = -1 if self.are_labels else 0

        segmented_arrays = np.array([
            [
                row[slice_start:] for row in X if row[self.segment_col] == segment
            ] for segment in set(X[:, self.segment_col])
        ])

        if len(segmented_arrays.shape) == 1 and type(segmented_arrays[0]) == list:
            raise(Exception('segmented_arrays is an array of {} lists, '.format(len(segmented_arrays)) +
                            'not a 3D matrix. Make sure that the set of segment values ' +
                            '(count = {}) is correct and '.format(len(segmented_arrays)) +
                            'divides evenly into total rows ({})'.format(len(X))))

        return segmented_arrays


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Uses LabelEncoder to encode given category/object columns.

    Parameters
    ----------
        columns : {array-like}
            List of names of columns to encode with LabelEncoder.
    """

    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
            X : {DataFrame}, shape = [n_observations, n_features].
            Samples

        Returns
        -------
            {DataFrame}, shape = [n_observations, n_features]
                Samples with category columns encoded.
        """

        encoder = LabelEncoder()
        df = X.copy()
        for column in self.columns:
            df.loc[:, column] = encoder.fit_transform(df[column])

        return df


class TimeStepOneHotEncoder(BaseEstimator, TransformerMixin):
        """
        Uses OneHoteEncoder to encode labels along axis 2.

        Parameters
        ----------
            sparse : {boolean}
                Parameter for OneHoteEncoder. Determines if returned matrix is sparse.
        """

        def __init__(self, sparse=False):
            self.sparse = sparse

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            """
            Parameters
            ----------
                X : {array-like}, shape = [n_observations, n_steps, 1]
                Samples

            Returns
            -------
                {array-like}, shape = [n_observations, n_steps, n_categories]
                    (n_categories -1 if sparse=True)
                    Samples with axis 2 expanded to number of categories
            """

            # Collapse arrays and get unique category values
            n_categories = len(set(np.concatenate(np.concatenate(X))))

            return OneHotEncoder(
                sparse=self.sparse
            ).fit_transform(
                X[:, :, 0]
            ).reshape(
                -1, X.shape[1], n_categories
            )
