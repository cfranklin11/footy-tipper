from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


UNSHIFTED_COLS = [
    'team',
    'oppo_team',
    'home_team',
    'year',
    'round_number',
    'win_odds',
    'point_spread',
]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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

    def __init__(self, include=None, exclude=None):
        self.include = include or '.*' if exclude is None else include
        self.exclude = exclude

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
            return X.filter(regex=self.include).values

        if self.exclude is not None:
            drop_cols = X.filter(regex=self.exclude).columns.values
            return X.drop(drop_cols, axis=1).values

        return X.values


class CumulativeFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Reshapes a stacked DataFrame into an array of matrices segmented by team and
    adds features with cumulative sums and rolling averages.

    Parameters
    ----------
        window_size : {int}
            Size of the rolling window for calculating average score, average oppo score,
            and average percentage.
        k_factor : {int}
            Rate of change for elo rating after each match.
    """

    def __init__(self, window_size=23, k_factor=32, unshifted_cols=UNSHIFTED_COLS):
        self.window_size = window_size
        self.k_factor = k_factor
        self.unshifted_cols = unshifted_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
            X : {DataFrame}, shape = [n_observations, n_features]
                Samples.

        Returns
        -------
            {DataFrame}, shape = [n_observations, n_features + n_cumulative_features]
                Samples with additional features, segmented by team.
        """

        unique_teams = X.index.get_level_values(0).drop_duplicates()

        team_df = pd.concat(
            [self.__create_team_df(X, team) for team in unique_teams],
            axis=0
        )

        elo_df = self.__create_elo_df(team_df)

        return pd.concat([team_df, elo_df], axis=1)

    def __create_team_df(self, X, team):
        filtered_df = X.xs(
            team, level=0, drop_level=False
        )
        shifted_df = filtered_df.drop(
            self.unshifted_cols, axis=1
        ).sort_index(
            ascending=True
        ).shift(
        ).fillna(
            0
        )
        shifted_df.columns = 'last_' + shifted_df.columns.values

        return pd.concat(
            [filtered_df[self.unshifted_cols], shifted_df],
            axis=1
        ).assign(
            last_win=(shifted_df['last_score'] > shifted_df['last_oppo_score']).astype(int),
            rolling_avg_score=lambda x: self.__rolling_mean(x['last_score']),
            rolling_avg_oppo_score=lambda x: self.__rolling_mean(x['last_oppo_score']),
            rolling_percent=lambda x: (
                self.__rolling_mean(x['last_score']) / self.__rolling_mean(x['last_oppo_score'])
            ).fillna(
                0
            )
        )

    def __rolling_mean(self, col):
        return col.rolling(self.window_size, min_periods=1).mean().fillna(0)

    def __create_elo_df(self, df):
        team_df = df.loc[
            :, ['oppo_team', 'last_win', 'last_score']
        ].pivot_table(
            index=['year', 'round_number'],
            values=['oppo_team', 'last_win', 'last_score'],
            columns='team',
            # Got this aggfunc from:
            # https://medium.com/@enricobergamini/creating-non-numeric-pivot-tables-with-python-pandas-7aa9dfd788a7
            aggfunc=lambda x: ' '.join(str(v) for v in x)
        )
        unique_teams = team_df.columns.get_level_values(1).drop_duplicates()
        elo_dict = {team: [] for team in unique_teams}
        row_indices = list(zip(team_df.index.get_level_values(0), team_df.index.get_level_values(1)))

        for idx, row in enumerate(row_indices):
            for team in unique_teams:
                elo_dict[team].append(self.__calculate_elo(idx, row, row_indices, team_df, team, elo_dict))

        elo_df = pd.DataFrame(
            elo_dict
        ).assign(
            year=team_df.index.get_level_values(0), round_number=team_df.index.get_level_values(1)
        )

        return pd.melt(
            elo_df,
            var_name='team',
            value_name='elo_rating',
            id_vars=['year', 'round_number']
        ).set_index(
            ['team', 'year', 'round_number']
        )

    def __calculate_elo(self, idx, row, row_indices, team_df, team, elo_dict):
        this_round = team_df.loc[row, :]
        last_row = row_indices[idx - 1] if idx > 0 else None
        last_round = None if last_row is None else team_df.loc[last_row]

        if last_round is None:
            # Start teams out at 1500 unless they don't exist in the first available season
            if team_df.loc[(row[0], slice(None)), ('last_score', team)].sum() > 0:
                return 1500
            else:
                return 0

        this_team = this_round[(slice(None), team)]
        last_team = last_round[(slice(None), team)]

        last_oppo_team_name = last_team['oppo_team']
        last_oppo_team = None if last_oppo_team_name == '0' else last_round[(slice(None), last_oppo_team_name)]

        last_team_elo = self.__calculate_last_team_elo(elo_dict, team, idx, team_df, row, last_row)

        # If last week was a bye, just keep the same elo
        if last_oppo_team is None:
            return last_team_elo

        last_oppo_team_elo = self.__calculate_last_team_elo(elo_dict, last_oppo_team_name, idx, team_df, row, last_row)

        team_r = 10**(last_team_elo / 400)
        oppo_r = 10**(last_oppo_team_elo / 400)
        team_e = team_r / (team_r + oppo_r)
        team_s = int(this_team['last_win'])
        team_elo = int(last_team_elo + self.k_factor * (team_s - team_e))

        return team_elo

    def __calculate_last_team_elo(self, elo_dict, team, idx, team_df, row, last_row):
        last_team_elo = elo_dict[team][idx - 1]
        # Revert to the mean by 25% at the start of every season
        if row[1] == 1:
            if last_team_elo != 0:
                last_team_elo = last_team_elo * 0.75 + 1500 * 0.25
            # If team is new, they start at 1300
            if (
                team_df.loc[(last_row[0], slice(None)), ('last_score', team)].sum() == 0 and
                team_df.loc[(row[0], slice(None)), ('last_score', team)].sum() != 0
            ):
                last_team_elo = 1300

        return last_team_elo


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

    def __init__(self, n_steps=None, are_labels=False, segment_col=0):
        if n_steps is None:
            raise 'n_steps is None. Be sure to explicitly set n_steps to avoid data shape errors.'

        self.n_steps = n_steps
        self.are_labels = are_labels
        self.segment_col = segment_col

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

        # [n_segments, n_observations / n_segments, n_features]
        X_segmented = self.__segment_arrays(X)

        return np.array(
            [
                # Shift by time steps: [n_steps, n_segments, n_observations / n_segments, n_features]
                X_segmented[:, self.n_steps - step_n:-step_n or None, :] for step_n in range(self.n_steps)
            ]
        ).reshape(
            # Reshape to eliminate segmentation: [n_steps, n_observations, n_features]
            self.n_steps, -1, X_segmented.shape[-1]
        ).transpose(
            # Transpose into shape for RNN: [n_observations, n_steps, n_features]
            1, 0, 2
        )

    def __segment_arrays(self, X):
        return np.array([
            [
                # If input is labels instead of features, ignore first column,
                # which is only there for segmenting purposes
                row[int(self.are_labels):] for row in X if row[self.segment_col] == segment
            ] for segment in set(X[:, self.segment_col])
        ])


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
                {array-like}, shape = [n_observations, n_steps, n_categories] (n_categories -1 if sparse=True)
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


class TimeStepDFCreator(BaseEstimator, TransformerMixin):
    """
    Reshapes time series DataFrame to add features from previous time steps
    (t - 1, t - 2, ..., t - n) as time-step features to the observation for
    time step t. This is in anticipation of reshaping the DataFrame into a 3D
    Numpy array for an RNN.

    Parameters
    ----------
        n_steps : {int}
        Number of steps back in time added to each observation.

        window_size : {int}
        Size of the rolling window for calculating average score, average oppo score,
        and average percentage.
    """
    def __init__(self, n_steps=None):
        if n_steps is None:
            raise 'n_steps is None. Be sure to explicitly set n_steps to avoid data shape errors.'

        self.n_steps = n_steps

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
            X : {DataFrame}, shape = [n_observations, n_features + 1].
            Samples

        Returns
        -------
            {DataFrame}, shape = [n_observations, n_features * n_steps]
            Samples with added time-step features
        """

        return self.__create_time_step_df(
            X,
            self.n_steps
        )

    def __create_time_step_df(self, df, n_steps):
        unique_teams = df.index.get_level_values(0).drop_duplicates()
        team_dfs = [
            self.__create_team_df(
                df.xs(team, level=0, drop_level=False), n_steps
            ) for team in unique_teams
        ]

        return pd.concat(team_dfs, axis=0)

    def __create_team_df(self, df, n_steps):
        step_dfs = [self.__create_team_time_step_df(df, step_n) for step_n in range(n_steps + 1)]

        return pd.concat(step_dfs, axis=1).dropna()

    def __create_team_time_step_df(self, df, step_n):
        stepped_df = df.shift(step_n)
        stepped_df.columns = stepped_df.columns.values + f'_t{step_n}'
        return stepped_df
