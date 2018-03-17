import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import dateutil
import re
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from sklearn.base import BaseEstimator, TransformerMixin

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.models import Match


ROW_INDEXES = ['team', 'year', 'round_number']
DIGITS = re.compile('round\s+(\d+)$', flags=re.I)
MAX_REG_ROUND = 24
QUALIFYING = re.compile('qualifying', flags=re.I)
ELIMINATION = re.compile('elimination', flags=re.I)
SEMI = re.compile('semi', flags=re.I)
PRELIMINARY = re.compile('preliminary', flags=re.I)
GRAND = re.compile('grand', flags=re.I)
UNSHIFTED_COLS = ['team', 'oppo_team', 'last_finals_reached', 'venue', 'home_team',
                  'year', 'round_number', 'win_odds', 'point_spread']


class ModelData():
    def __init__(self, db_url, train=False):
        if train:
            db_df = DBData(db_url).data()
            min_year = db_df['year'].min()
            # Have to read some data off a CSV due to DB size restrictions for a free app
            # on Heroku
            csv_df = CSVData(os.path.join(project_path, 'data')).data(max_year=min_year - 1)
            full_df = csv_df.append(db_df).sort_index().fillna(0)
        else:
            this_year = datetime.now().year
            # TimeStepVotingClassifier expects data from this year & previous year,
            # and reshapes the data per n_steps param of each model
            full_df = DBData(db_url).data(year_range=(this_year - 1, this_year)).sort_index()

        self.df = CumulativeFeatureBuilder().transform(full_df.drop('win', axis=1), y=full_df['win'])

    def data(self):
        return self.df

    def prediction_data(self):
        X = self.__team_df(self.df).drop('win', axis=1)
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

        slice_end = -blank_count or None

        return team_df.iloc[:slice_end, :]


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

    def data(self, max_year=2000):
        match_df = self.__create_match_df(os.path.join(self.data_directory, 'ft_match_list.csv'))
        betting_df = self.__create_betting_df(os.path.join(self.data_directory, 'afl_betting.csv'))
        merged_df = match_df.merge(betting_df, on=['team', 'full_date'], how='left').fillna(0)

        clean_df = self.clean_match_df(merged_df)

        return clean_df[clean_df['year'] <= max_year].drop('full_date', axis=1)

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

    def data(self, year_range=(0, None)):
        engine = create_engine(self.db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        data = self.__fetch_data(session, year_range)
        df = self.__create_df(data)
        clean_df = self.clean_match_df(df)

        session.close()

        return clean_df.drop('full_date', axis=1)

    def __fetch_data(self, session, year_range):
        if year_range[1] is None:
            date_filter = and_(Match.date >= datetime(int(year_range[0]), 1, 1))
        else:
            date_filter = and_(
                Match.date >= datetime(int(year_range[0]), 1, 1),
                Match.date <= datetime(int(year_range[1]) + 1, 1, 1)
            )

        return session.query(Match).filter(date_filter).all()

    def __create_df(self, data):
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

    def __init__(self, window_size=23, k_factor=20, unshifted_cols=UNSHIFTED_COLS):
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

        if y is None:
            df = X
        else:
            df = pd.concat([X, y], axis=1)
            self.unshifted_cols.append('win')

        team_df = pd.concat(
            [self.__create_team_df(df, team) for team in unique_teams],
            axis=0
        )
        elo_df = self.__create_elo_df(team_df)
        concat_df = pd.concat([team_df, elo_df], axis=1)

        return self.__zero_out_filler_rounds(concat_df)

    def __zero_out_filler_rounds(self, df):
        string_cols = df.select_dtypes([object]).columns.values
        real_rounds = df[df['oppo_team'] != '0']
        filler_rounds = df[df['oppo_team'] == '0']

        filler_rounds.loc[:, :] = 0
        filler_rounds.loc[:, string_cols] = filler_rounds[string_cols].astype(str)
        filler_rounds.loc[:, ['team', 'year', 'round_number']] = np.array(
            # Fill in index columns with indexes for reset_index/set_index in later transformations
            [filler_rounds.index.get_level_values(level) for level in range(len(filler_rounds.index.levels))]
        ).transpose(1, 0)

        return pd.concat([real_rounds, filler_rounds]).sort_index()

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
        last_win_column = (shifted_df['last_score'] > shifted_df['last_oppo_score']).astype(int)

        team_df = pd.concat(
            [filtered_df[self.unshifted_cols], shifted_df],
            axis=1
        ).assign(
            last_win=last_win_column,
            rolling_avg_score=lambda x: self.__rolling_mean(x['last_score']),
            rolling_avg_oppo_score=lambda x: self.__rolling_mean(x['last_oppo_score']),
            rolling_percent=lambda x: (
                self.__rolling_mean(x['last_score']) / self.__rolling_mean(x['last_oppo_score'])
            ).fillna(
                0
            ),
            rolling_win_percent=self.__rolling_mean(last_win_column),
            cum_score=lambda x: x.groupby(level=[0, 1])['last_score'].cumsum(),
            cum_oppo_score=lambda x: x.groupby(level=[0, 1])['last_oppo_score'].cumsum()
        )
        team_df['cum_wins'] = team_df.groupby(level=[0, 1])['last_win'].cumsum()
        team_df['cum_percent'] = (team_df['cum_score'] / team_df['cum_oppo_score']).fillna(0)

        return team_df

    def __rolling_mean(self, col):
        return col.rolling(self.window_size, min_periods=1).mean().fillna(0)

    def __create_elo_df(self, df):
        team_df = df.loc[
            :, ['oppo_team', 'last_win', 'last_score']
        ].pivot_table(
            index=['year', 'round_number'],
            values=['oppo_team', 'last_win', 'last_score'],
            columns='team',
            # Got this aggfunc for using string column in values from:
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
        if last_oppo_team_name == '0':
            last_oppo_team = None
        else:
            last_oppo_team = last_round[(slice(None), last_oppo_team_name)]

        last_team_elo = self.__calculate_last_team_elo(elo_dict, team, idx, team_df, row, last_row)

        # If last week was a bye, just keep the same elo
        if last_oppo_team is None:
            return last_team_elo

        last_oppo_team_elo = self.__calculate_last_team_elo(
            elo_dict, last_oppo_team_name, idx, team_df, row, last_row
        )

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
