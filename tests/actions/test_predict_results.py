import unittest
import os
import sys
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.estimators.estimator import Estimator


ROW_INDEXES = ['team', 'year', 'round_number']


@unittest.skip('Estimator pipeline transformers are too dependent on each other ' +
               'and brittle regarding data changes, making fixture data too difficult to maintain')
class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.described_class = Estimator
        self.X = (pd.read_csv(os.path.join(project_path, 'tests/fixtures/X.csv'))
                    .set_index(ROW_INDEXES, drop=False))
        self.y = (pd.read_csv(os.path.join(project_path, 'tests/fixtures/y.csv'))
                    .set_index(ROW_INDEXES, drop=False))

    def test_predict(self):
        teams = self.X['team'].drop_duplicates()
        predictions = self.described_class().predict(self.X, self.y)

        self.assertIsInstance(predictions, list)
        self.assertIsInstance(predictions[0], dict)
        self.assertIn('round_number', predictions[0].keys())
        self.assertIn('home_team', predictions[0].keys())
        self.assertIn('away_team', predictions[0].keys())
        self.assertIn('home_win_predicted', predictions[0].keys())
        self.assertEqual(len(teams) / 2, len(predictions))
