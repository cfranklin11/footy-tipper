import os
import sys
import re
from datetime import datetime
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.models import Match

ROW_INDEXES = ['team', 'year', 'round_number']
DIGITS = re.compile('\d\d?$')


class MatchData():
    def __init__(self, db_url):
        self.db_url = db_url

    def data(self, year_range=(2012, 2016)):
        engine = create_engine(self.db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        data = self.__fetch_data(session, year_range)
        session.close()

        df = self.__create_match_df(data)
        return self.__clean_match_df(df)

    def __fetch_data(self, session, year_range):
        return session.query(
            Match
        ).filter(
            and_(
                Match.date > datetime(year_range[0], 1, 1),
                Match.date < datetime(year_range[1] + 1, 1, 1)
            )
        ).all()

    def __create_match_df(self, data):
        df_dict = {
            'year': [],
            'round_number': [],
            'team': [],
            'oppo_team': [],
            'home_team': [],
            'score': [],
            'oppo_score': [],
            'win_odds': [],
            'point_spread': []
        }

        for match in data:
            # Skipping finals rounds to keep data simple & consistent
            if DIGITS.search(match.season_round) is None:
                continue

            round_number = int(DIGITS.search(match.season_round)[0])

            df_dict['year'].extend([match.date.year, match.date.year])
            df_dict['round_number'].extend([round_number, round_number])
            df_dict['team'].extend([match.home_team.name, match.away_team.name])
            df_dict['oppo_team'].extend([match.away_team.name, match.home_team.name])
            df_dict['home_team'].extend([1, 0])
            df_dict['score'].extend([match.home_score, match.away_score])
            df_dict['oppo_score'].extend([match.away_score, match.home_score])
            df_dict['win_odds'].extend([match.home_betting_odds.win_odds, match.away_betting_odds.win_odds])
            df_dict['point_spread'].extend([match.home_betting_odds.point_spread, match.away_betting_odds.point_spread])

        return pd.DataFrame(df_dict).set_index(ROW_INDEXES, drop=False)

    def __clean_match_df(self, df):
        # Tied finals are replayed, resulting in duplicate team/year/round combos.
        # Dropping all but the last to get rid of ties, because it's easier than incorporating
        # them into the data.
        duplicate_indices = df.index.duplicated(keep='last')
        match_df = df[
            np.invert(duplicate_indices)
        ].dropna(
        ).assign(
            win=(df['score'] > df['oppo_score'])
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

        match_df.loc[:, ROW_INDEXES] = np.array(
            # Fill in index columns with indexes for reset_index/set_index in later steps
            [match_df.index.get_level_values(level) for level in range(len(match_df.index.levels))]
        ).transpose(
            1, 0
        )

        return match_df.sort_index()
