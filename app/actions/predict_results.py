import os
import sys
import pandas as pd
from sklearn.externals import joblib

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.model.preprocessing import TimeStepDFCreator


ROW_INDEXES = ['team', 'year', 'round_number']


class MLModel():
    def predict(self, X, y):
        estimator = joblib.load(os.path.join(project_path, 'app/estimators/TimeStepVotingClassifier.pkl'))

        y_pred_proba = estimator.predict_proba(X)
        pred_home_win = self.__compare_teams(X, y, y_pred_proba)
        pred_df = pd.concat(
            [X[['round_number', 'team', 'oppo_team']], pred_home_win], axis=1
        ).dropna()
        pred_df.columns = ['round_number', 'home_team', 'away_team', 'home_win_predicted']

        return pred_df.to_dict('records')

    def __compare_teams(self, X, y, y_pred_proba):
        y_win_proba = y_pred_proba[:, 1]
        column_suffix = '_t0'
        row_indexes = [f'{idx}{column_suffix}' for idx in ROW_INDEXES]

        X_test_transformed = TimeStepDFCreator(n_steps=self.n_steps).fit_transform(X)
        y_test_transformed = TimeStepDFCreator(n_steps=self.n_steps).fit_transform(y)['win_t0']

        win_df = X_test_transformed.assign(
            win=y_test_transformed,
            win_proba=y_win_proba
        )

        home_df = win_df[win_df[f'home_team{column_suffix}'] == 1]

        away_df = win_df[
            # On teams' bye weeks, oppo_team = '0'
            ((win_df[f'home_team{column_suffix}'] == 0) &
             (win_df[f'oppo_team{column_suffix}'] != '0'))
        ][
            ['win_proba', f'oppo_team{column_suffix}'] + row_indexes[1:]
        ].set_index(
            [f'oppo_team{column_suffix}'] + row_indexes[1:]
        )
        away_df.columns = 'away_' + away_df.columns.values
        away_df.index.rename('team', level=0, inplace=True)

        match_df = pd.concat([home_df, away_df], axis=1)

        # Compare home & away win probabilities to get predicted winner
        return (match_df['win_proba'] > match_df['away_win_proba']).apply(int).rename('home_win')
