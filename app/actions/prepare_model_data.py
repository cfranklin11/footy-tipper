import os
import sys
from datetime import datetime
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.actions.fetch_data import DBData, CSVData


class ModelData():
    def __init__(self, db_url, n_steps, train=False):
        self.n_steps = n_steps

        if train:
            db_df = DBData(db_url).data()
            min_year = db_df['year'].min()
            # Have to read some data off a CSV due to DB size restrictions for a free app
            # on Heroku
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
        # (All real matches should have a venue)
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
        slice_end = -blank_count or None

        return team_df.iloc[slice_start:slice_end, :]
