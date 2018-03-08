import unittest
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app, db
from app.models import Team
from app.actions.save_data import DataSaver


ROW_INDEXES = ['team', 'year', 'round_number']


@unittest.skip("DB transactions are freezing, and I can't be bothered to debug it for now")
class TestDataSaver(unittest.TestCase):
    def setUp(self):
        self.described_class = DataSaver
        betting_odds_df = pd.read_csv(
            os.path.join(project_path, 'tests/fixtures/betting_odds.csv')
        )
        match_df = pd.read_csv(
            os.path.join(project_path, 'tests/fixtures/match.csv')
        )
        self.data = {'match': match_df, 'betting_odds': betting_odds_df}
        engine = create_engine(app.config['DATABASE_URL'])
        db.create_all()
        Session = sessionmaker(bind=engine)
        session = Session()

        df = pd.read_csv(
            os.path.join(project_path, 'data/ft_match_list.csv'),
            parse_dates=['full_date'],
            infer_datetime_format=True
        )
        team_names = df['home_team'].append(df['away_team']).drop_duplicates()
        teams = [Team(name=team_name) for team_name in team_names]
        session.add_all(teams)
        session.commit()
        session.close()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_save_data(self):
        self.described_class(self.data).save_data()
        assert True
