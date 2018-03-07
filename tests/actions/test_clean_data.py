import unittest
import os
import sys
import json
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.actions.clean_data import DataCleaner


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.described_class = DataCleaner

    def test_data(self):
        with open(os.path.join(project_path, 'tests/fixtures/scraped_data.json'), 'r') as f:
            page_data = json.load(f)

        data = self.described_class(page_data).data()

        self.assertEqual(type(data), dict)
        self.assertEqual(len(data.keys()), 2)
        self.assertIn('match', data.keys())
        self.assertIn('betting_odds', data.keys())

        self.assertIsInstance(data['match'], pd.core.frame.DataFrame)
        self.assertIsInstance(data['betting_odds'], pd.core.frame.DataFrame)
        self.assertFalse(data['match'].isnull().any().any())
        self.assertFalse(data['betting_odds'].isnull().any().any())
