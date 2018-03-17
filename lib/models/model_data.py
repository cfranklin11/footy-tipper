import pandas as pd
import numpy as np


class TimeStepDataCreator():
    def _train_start(self, years):
        # Actual training data start one season after min in order to permit
        # lead-in rounds, so the resulting data will have the same number of rows
        # regardless of the value of n_steps
        return years.min() + 1

    def _team_df(self, df, years, start=None):
        unique_teams = df.index.get_level_values(0).drop_duplicates()

        return pd.concat(
            [
                df.xs(
                    [team, slice(*years)], level=[0, 1], drop_level=False
                    # Resetting index, because pandas flips out if I try to slice
                    # with iloc while the df has a multiindex
                ).sort_index(
                ).reset_index(
                    drop=True
                ).iloc[
                    start:, :
                ] for team in unique_teams
            ]
        ).set_index(
            ['team', 'year', 'round_number'], drop=False
        ).sort_index()

    def _year_ranges(self, train_start):
        return (train_start - 1, train_start - 1), (train_start, None)

    def _X_train(self, X, train_start=None):
        if train_start is None:
            # Subtract one extra from start to include current round + n_steps of previous rounds
            return self._team_df(X, (None, None), start=-self.n_steps - 1)

        train_lead_in_years, train_years = self._year_ranges(train_start)

        X_train_lead_in = self._team_df(X, train_lead_in_years, start=-self.n_steps)
        X_train_body = X.xs(slice(*train_years), level=1, drop_level=False)

        return pd.concat([X_train_lead_in, X_train_body]).sort_index()

    def _model_config(self, X):
        return {
            'n_teams': len(X['team'].drop_duplicates()),
            'n_train_years': len(X.index.get_level_values(1).drop_duplicates()),
            'n_venues': len(X['venue'].drop_duplicates()),
            # Add 1 to categories for last_finals_reached
            'n_categories': len(X.select_dtypes(exclude=[np.number]).columns) + 1,
            'n_features': len(X.columns)
        }


class FitDataCreator(TimeStepDataCreator):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def data(self, X, y):
        unique_years = X.index.get_level_values(1).drop_duplicates()

        if len(unique_years) < 2:
            raise(Exception(f'There are only {len(unique_years)} years worth ' +
                            'of data. Need at least 2 to allow for time-step reshaping.'))

        train_start = self._train_start(unique_years)

        return (
            self._X_train(X, train_start=train_start),
            self.__y_train(y, train_start),
            self._model_config(X)
        )

    def __y_train(self, y, train_start):
        train_lead_in_years, train_years = self._year_ranges(train_start)
        y_train_lead_in = self._team_df(y, train_lead_in_years, start=-self.n_steps)
        y_train_body = y.xs(slice(*train_years), level=1, drop_level=False)

        return pd.concat([y_train_lead_in, y_train_body])[['team', 'win']].sort_index()


class PredictDataCreator(TimeStepDataCreator):
    def __init__(self, n_steps, full_year=True):
        self.n_steps = n_steps
        self.full_year = full_year

    def data(self, X):
            # Actual training data start one season after min in order to permit
            # lead-in rounds, so the resulting data will have the same number of rows
            # regardless of the value of n_steps
            unique_years = X.index.get_level_values(1).drop_duplicates()

            if len(unique_years) < 2:
                raise(Exception(f'There are only {len(unique_years)} years worth ' +
                                'of data. Need at least 2 to allow for time-step reshaping.'))

            train_start = self._train_start(unique_years) if self.full_year else None

            return self._X_train(X, train_start=train_start), self._model_config(X)
