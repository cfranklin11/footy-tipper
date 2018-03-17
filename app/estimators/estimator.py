import os
import sys
import pandas as pd
from sklearn.externals import joblib

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)


ROW_INDEXES = ['team', 'year', 'round_number']


class Estimator():
    def __init__(self, estimator_path='app/estimators/TimeStepVotingClassifier_2018-03-17_1965_to_2017.pkl'):
        self.estimator_path = estimator_path

    def predict(self, X, y):
        estimator = joblib.load(os.path.join(project_path, self.estimator_path))

        y_pred_proba = estimator.predict_proba(X)
        pred_home_win = self.__compare_teams(X, y, y_pred_proba)
        pred_df = pd.concat(
            [X[['round_number', 'team', 'oppo_team']], pred_home_win], axis=1
        ).dropna()
        pred_df.columns = ['round_number', 'home_team', 'away_team', 'home_win_predicted']

        return pred_df.to_dict('records')

    def __compare_teams(self, X, y, y_pred_proba):
        X_years = X['year'].drop_duplicates()

        # If X & y have a lead-in year or lead-in rounds, drop the first year, so they line up
        # with y_pred
        if len(X) > len(y_pred_proba) and len(X_years) > 1:
            min_test_year = min(X_years) + 1
            X_test = X.xs(slice(min_test_year, None), level=1, drop_level=False)
            y_test = y.xs(slice(min_test_year, None), level=1, drop_level=False)
        else:
            X_test = X
            y_test = y

        y_win = y_test['win'] if len(y_test.shape) > 1 else y_test
        y_win_proba = y_pred_proba[:, 1]

        win_df = X_test.assign(
            win=y_win,
            win_proba=y_win_proba
        )

        home_df = win_df[win_df['home_team'] == 1]

        # On teams' bye weeks, oppo_team = '0'
        away_df = (win_df[(win_df['home_team'] == 0) & (win_df['oppo_team'] != '0')]
                   [['win_proba', 'oppo_team'] + ROW_INDEXES[1:]]
                   .set_index(['oppo_team'] + ROW_INDEXES[1:]))
        away_df.columns = 'away_' + away_df.columns.values
        away_df.index.rename('team', level=0, inplace=True)

        match_df = pd.concat([home_df, away_df], axis=1)

        # Compare home & away win probabilities to get predicted winner
        return (match_df['win_proba'] > match_df['away_win_proba']).apply(int).rename('home_win')
