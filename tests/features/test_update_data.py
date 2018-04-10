import unittest
import os
import sys
import re
from datetime import datetime

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app, db
from app.models import Match, BettingOdds
from db import seeds


SQLITE = re.compile('sqlite')


class TestUpdateData(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

        if (
            SQLITE.search(app.config['SQLALCHEMY_DATABASE_URI']) is None or
            SQLITE.search(app.config['DATABASE_URL']) is None
        ):
            raise(Exception("Database URL isn't set to temporary SQLite DB. Please check app config"))

        db.create_all()
        seeds.main()

    def tearDown(self):
        db.drop_all()
        os.remove(os.path.join(project_path, 'test.db'))

    def test_update_data(self):
        old_match_count = db.session.query(Match).count()
        old_betting_odds_count = db.session.query(BettingOdds).count()

        self.app.post('/update_data?password={}'.format(app.config['PASSWORD']))

        new_match_count = db.session.query(Match).count()
        new_betting_odds_count = db.session.query(BettingOdds).count()

        current_year = datetime.now().year
        current_year_match_count = (db.session
                                      .query(Match)
                                      .filter(Match.date > datetime(current_year, 1, 1))
                                      .count())

        # DataSaver updates all missing match data, but only this week's betting odds
        self.assertEqual(old_match_count + current_year_match_count, new_match_count)
        self.assertEqual(old_betting_odds_count + 18, new_betting_odds_count)

        self.app.post('/update_data?password={}'.format(app.config['PASSWORD']))

        new_new_match_count = db.session.query(Match).count()
        new_new_betting_odds_count = db.session.query(BettingOdds).count()

        # With recently updated DB, no new records should be saved
        self.assertEqual(new_match_count, new_new_match_count)
        self.assertEqual(new_betting_odds_count, new_new_betting_odds_count)
