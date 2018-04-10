import unittest
import os
import sys
import re
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app, db
from app.models import Team, Match, BettingOdds
from app.actions.save_data import DataSaver


SQLITE = re.compile('sqlite')


@unittest.skip('Mocking datetime.now() is stupidly hard in Python, and DataSaver checks ' +
               'current year to determine what data to save, so skipping for now.')
class TestDataSaver(unittest.TestCase):
    def setUp(self):
        self.described_class = DataSaver
        betting_odds_df = pd.read_csv(
            os.path.join(project_path, 'tests/fixtures/betting_odds.csv'), parse_dates=['date']
        )
        match_df = pd.read_csv(
            os.path.join(project_path, 'tests/fixtures/match.csv'), parse_dates=['date']
        )
        self.data = {'match': match_df, 'betting_odds': betting_odds_df}

        if (
            SQLITE.search(app.config['SQLALCHEMY_DATABASE_URI']) is None or
            SQLITE.search(app.config['DATABASE_URL']) is None
        ):
            raise(Exception("Database URL isn't set to temporary SQLite DB. Please check app config"))

        db.create_all()

        df = pd.read_csv(
            os.path.join(project_path, 'data/ft_match_list.csv'),
            parse_dates=['full_date'],
            infer_datetime_format=True
        )
        team_names = df['home_team'].append(df['away_team']).drop_duplicates()
        teams = [Team(name=team_name) for team_name in team_names]
        db.session.add_all(teams)
        db.session.commit()
        db.session.close()

    def tearDown(self):
        db.drop_all()
        os.remove(os.path.join(project_path, 'test.db'))

    def test_save_data(self):
        self.described_class(self.data).save_data()

        self.assertEqual(len(self.data['match']), db.session.query(Match).count())
        self.assertEqual(len(self.data['betting_odds']), db.session.query(BettingOdds).count())

    def test_save_data_with_update(self):
        match_length = len(self.data['match'])
        betting_length = len(self.data['betting_odds'])
        partial_data = {
            'match': self.data['match'].iloc[:int(match_length / 2), :],
            'betting_odds': self.data['betting_odds'].iloc[:int(betting_length / 2), :]
        }
        self.described_class(partial_data).save_data()
        self.described_class(self.data).save_data()

        self.assertEqual(len(self.data['match']), db.session.query(Match).count())
        self.assertEqual(len(self.data['betting_odds']), db.session.query(BettingOdds).count())
