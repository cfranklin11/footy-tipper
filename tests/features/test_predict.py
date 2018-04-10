import unittest
import os
import sys
import re
import json

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app, db
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
        response = self.app.post('/predict?password={}'.format(app.config['PASSWORD']))
        decoded_response = json.loads(response.data.decode('utf8'))

        self.assertEqual(len(decoded_response), 9)

        prediction_keys = set(['round_number', 'home_team', 'away_team', 'home_win_predicted', 'full_date'])
        response_key_lists = [prediction.keys() for prediction in decoded_response]
        response_keys = set([response_key for key_list in response_key_lists for response_key in key_list])

        self.assertEqual(prediction_keys, response_keys)
