import os
import sys
import re
from datetime import datetime
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import dateutil

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.models import Match
from lib.model.preprocessing import TimeStepDFCreator

ROW_INDEXES = ['team', 'year', 'round_number']
DIGITS = re.compile('round\s+(\d+)$', flags=re.I)
QUALIFYING = re.compile('qualifying', flags=re.I)
ELIMINATION = re.compile('elimination', flags=re.I)
SEMI = re.compile('semi', flags=re.I)
PRELIMINARY = re.compile('preliminary', flags=re.I)
GRAND = re.compile('grand', flags=re.I)
MAX_REG_ROUND = 24


class MatchData():
    def clean_match_df(self, df):
        match_df = self.__drop_duplicate_indices(
            df.dropna(
            ).assign(
                win=(df['score'] > df['oppo_score']).astype(int),
                round_number=lambda x: x['season_round'].apply(self.__get_round_number)
            ).set_index(
                ROW_INDEXES, drop=False
            ).drop(
                'season_round', axis=1
            )
        )

        # Create finals category features per furthest round reached the previous year
        last_finals_reached = pd.DataFrame(
            match_df['round_number'].groupby(
                level=[0, 1]
            ).apply(
                lambda x: max(max(x) - MAX_REG_ROUND, 0)
            ).groupby(
                level=[0]
            ).shift(
            ).fillna(
                0
            ).rename(
                'last_finals_reached'
            )
        ).reset_index()

        match_df = match_df.merge(
            last_finals_reached, on=['team', 'year'], how='left'
        ).set_index(
            ROW_INDEXES, drop=False
        ).unstack(
            # Unstack year & round_number, fill, restack, then repeat with team & year to
            # make teams, years, and round_numbers consistent for all possible cross-sections
            # of the data
            ROW_INDEXES[1:]
        ).fillna(
            0
        ).stack(
            ROW_INDEXES[1:]
        ).unstack(
            ROW_INDEXES[:2]
        ).fillna(
            0
        ).stack(
            ROW_INDEXES[:2]
        ).reorder_levels(
            [1, 2, 0]
        ).sort_index()

        # stacked_df.loc[:, 'year'] = stacked_df['year'].astype(int)
        # stacked_df.loc[:, 'home_team'] = stacked_df['home_team'].astype(int)
        # stacked_df.loc[:, 'oppo_score'] = stacked_df['oppo_score'].astype(int)
        # stacked_df.loc[:, 'score'] = stacked_df['score'].astype(int)

        # Convert 0s in category columns to strings for later transformations
        string_cols = match_df.select_dtypes([object]).columns.values
        match_df.loc[:, string_cols] = match_df[string_cols].astype(str)

        match_df.loc[:, ROW_INDEXES] = np.array(
            # Fill in index columns with indexes for reset_index/set_index in later steps
            [match_df.index.get_level_values(level) for level in range(len(match_df.index.levels))]
        ).transpose(
            1, 0
        )

        return match_df[match_df['round_number'] <= MAX_REG_ROUND].sort_index()

    def __drop_duplicate_indices(self, df):
        # Tied finals are replayed, resulting in duplicate team/year/round combos.
        # Dropping all but the last to get rid of ties, because it's easier than incorporating
        # them into the data.
        duplicate_indices = df.index.duplicated(keep='last')
        return df[np.invert(duplicate_indices)].sort_index()

    def __get_round_number(self, x):
        digits = DIGITS.search(x)
        if digits is not None:
            return int(digits[1])
        if QUALIFYING.search(x) is not None:
            return 25
        if ELIMINATION.search(x) is not None:
            return 25
        if SEMI.search(x) is not None:
            return 26
        if PRELIMINARY.search(x) is not None:
            return 27
        if GRAND.search(x) is not None:
            return 28

        raise Exception("Round label doesn't match any known patterns")


class CSVData(MatchData):
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def data(self, max_year=1989):
        match_df = self.__create_match_df(os.path.join(self.data_directory, 'ft_match_list.csv'))
        betting_df = self.__create_betting_df(os.path.join(self.data_directory, 'afl_betting.csv'))
        merged_df = match_df.merge(betting_df, on=['team', 'full_date'], how='left').fillna(0)

        clean_df = self.clean_match_df(merged_df)

        return clean_df[clean_df['year'] <= max_year]

    def __create_match_df(self, file_path):
        df = pd.read_csv(
            file_path,
            parse_dates=[0],
            converters={
                'full_date': lambda x: dateutil.parser.parse(x).date()
            }
        ).assign(
            year=lambda x: x['full_date'].apply(lambda x: x.year)
        )

        return pd.DataFrame({
            'full_date': df['full_date'].append(df['full_date']).reset_index(drop=True),
            'season_round': df['season_round'].append(df['season_round']).reset_index(drop=True),
            'year': df['year'].append(df['year']).reset_index(drop=True),
            'team': df['home_team'].append(df['away_team']).reset_index(drop=True),
            'oppo_team': df['away_team'].append(df['home_team']).reset_index(drop=True),
            'home_team': np.append(np.ones(len(df)), np.zeros(len(df))),
            'score': df['home_score'].append(df['away_score']).reset_index(drop=True),
            'oppo_score': df['away_score'].append(df['home_score']).reset_index(drop=True),
            'venue': df['venue'].append(df['venue']).reset_index(drop=True)
        })

    # NOTE: Betting data is is stacked (each team/match combo is on a separate row)
    # by default, so doesn't require appending away rows to home rows like match data do
    def __create_betting_df(self, file_path):
        return pd.read_csv(
            file_path,
            parse_dates=[0],
            converters={
                'full_date': lambda x: dateutil.parser.parse(x).date()
            }
        ).loc[
            :, ['full_date', 'win_odds', 'point_spread', 'team']
        ]


class DBData(MatchData):
    def __init__(self, db_url):
        self.db_url = db_url

    def data(self, year_range=(1, 3000)):
        engine = create_engine(self.db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        data = self.__fetch_data(session, year_range)
        df = self.__create_match_df(data)
        clean_df = self.clean_match_df(df)

        session.close()

        return clean_df

    def __fetch_data(self, session, year_range):
        return session.query(
            Match
        ).filter(
            and_(
                Match.date >= datetime(int(year_range[0]), 1, 1),
                Match.date <= datetime(int(year_range[1]) + 1, 1, 1)
            )
        ).all()

    def __create_match_df(self, data):
        df_dict = {
            'year': [],
            'season_round': [],
            'team': [],
            'full_date': [],
            'oppo_team': [],
            'home_team': [],
            'venue': [],
            'score': [],
            'oppo_score': [],
            'win_odds': [],
            'point_spread': []
        }

        for match in data:
            df_dict['year'].extend([match.date.year, match.date.year])
            df_dict['season_round'].extend([match.season_round, match.season_round])
            df_dict['team'].extend([match.home_team.name, match.away_team.name])
            df_dict['full_date'].extend([match.date, match.date])
            df_dict['oppo_team'].extend([match.away_team.name, match.home_team.name])
            df_dict['venue'].extend([match.venue, match.venue])
            df_dict['home_team'].extend([1, 0])
            df_dict['score'].extend([match.home_score, match.away_score])
            df_dict['oppo_score'].extend([match.away_score, match.home_score])

            if match.home_betting_odds and match.away_betting_odds:
                df_dict['win_odds'].extend(
                    [match.home_betting_odds.win_odds, match.away_betting_odds.win_odds]
                )
                df_dict['point_spread'].extend(
                    [match.home_betting_odds.point_spread, match.away_betting_odds.point_spread]
                )
            else:
                df_dict['win_odds'].extend([0, 0])
                df_dict['point_spread'].extend([0, 0])

        return pd.DataFrame(df_dict)


class ModelData():
    def __init__(self, db_url, n_steps, train=False):
        self.n_steps = n_steps

        if train:
            db_df = DBData(db_url).data()
            min_year = db_df['year'].min()
            csv_df = CSVData(os.path.join(project_path, 'data')).data(max_year=min_year - 1)

            self.df = csv_df.append(db_df).sort_index().fillna(0).drop('full_date', axis=1)
        else:
            this_year = datetime.now().year
            # Reducing quantity of data without minimising it, because getting the exact
            # date range needed for predictions would be complicated & error prone
            self.df = DBData(db_url).data(year_range=(this_year - 1, this_year)).sort_index()

    def data(self):
        return self.df

    def prediction_data(self):
        X = self.__team_df(self.df).drop('win', axis=1).drop('full_date', axis=1)
        y = self.__team_df(
            self.df[['team', 'year', 'round_number', 'venue', 'win']]
        ).drop(
            ['year', 'round_number', 'venue'], axis=1
        )

        return X, y

    def __team_df(self, df):
        unique_teams = df.index.get_level_values(0).drop_duplicates()
        max_year = df['year'].max()
        unique_rounds = df.xs(max_year, level=1)['round_number'].drop_duplicates().values

        # Filling missing rounds to make sure all segments are the same shape
        # can lead to the current round being blank. We want to predict on the latest real round.
        blank_count = 0
        for round_number in unique_rounds[::-1]:
            round_venues = df.xs([max_year, round_number], level=[1, 2])['venue']

            if round_venues.apply(lambda x: x == '0').all():
                blank_count += 1
            else:
                break

        return pd.concat(
            [self.__get_latest_round(df, blank_count, team) for team in unique_teams]
        ).set_index(
            ['team', 'year', 'round_number'], drop=False
        ).sort_index()

    def __get_latest_round(self, df, blank_count, team):
        team_df = df.xs(
            team, level=0, drop_level=False
            # Resetting index, because pandas flips out if I try to slice
            # with iloc while the df has a multiindex
        ).sort_index(
        ).reset_index(
            drop=True
        )

        # Pad with earlier rounds per n_steps, so reshaping per time steps will work
        slice_start = -self.n_steps - (1 * (blank_count + 1))
        slice_end = blank_count or None

        return team_df.iloc[slice_start:slice_end, :]


class MLModel():
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def predict(self, X, y):
        X_pipeline = joblib.load(os.path.join(project_path, 'app/middleware/X_pipeline.pkl'))
        model = joblib.load(os.path.join(project_path, 'app/middleware/model.pkl'))

        X_transformed = X_pipeline.transform(X)

        y_pred_proba = model.predict_proba(X_transformed)
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
