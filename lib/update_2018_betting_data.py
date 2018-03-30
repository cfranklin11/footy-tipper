import os
import sys
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.actions.scrape_data import PageScraper
from app.actions.clean_data import DataCleaner
from app.actions.save_data import DataSaver


if __name__ == '__main__':
    # Update 2018 match data
    data = PageScraper().data()
    dfs = DataCleaner(data).data()
    DataSaver(dfs).save_data()

    # Update 2018 betting data as used for first 2 rounds
    df = pd.read_csv(os.path.join(project_path, 'data/afl_betting_2018.csv'), parse_dates=['date'])
    DataSaver({'betting_odds': df}).save_data()
