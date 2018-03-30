import os
import sys
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.actions.save_data import DataSaver


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(project_path, 'data/afl_betting_2018.csv'), parse_dates=['date'])
    DataSaver({'betting_odds': df}).save_data()
