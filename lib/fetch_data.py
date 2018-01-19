import os
import sys
from datetime import datetime
import csv
import requests
from bs4 import BeautifulSoup
import numpy as np


DOMAIN = 'https://www.footywire.com'
PATH = '/afl/footy/'
PAGES = [
    # Default year for data is most recent season. Append '?year=####' to get different years
    'ft_match_list',
    'afl_betting'
]


def get_betting_row(tr, season_year):
        table_row = list(tr.stripped_strings)

        if len(table_row) == 0:
            return []

        # First two columns in data rows have colspan="2", so empty cells need to be prepended
        # to every-other data row. There doesn't seem to be a good way of identifying these rows
        # apart from their length: 11 cells means the date is in the row, 9 means there's no date
        if len(table_row) == 9:
            return [season_year] + ([np.nan] * 2) + table_row

        return [season_year] + table_row


def betting_data(data_div, season_year):
    # afl_betting page nests the data table inside of an outer table
    data_table = data_div.find('table').find('table')

    if data_table is None:
        return None

    return [get_betting_row(tr, season_year) for tr in data_table.find_all('tr')]


def get_fixture_row(tr, season_year):
    table_row = list(tr.stripped_strings)

    if len(table_row) == 0:
        return []

    return [season_year] + table_row


def fixture_data(data_div, year):
    data_table = data_div.find('table')

    if data_table is None:
        return None

    return [get_fixture_row(tr, year) for tr in data_table.find_all('tr')]


def fetch_page_data(page_url, year):
    response = requests.get(page_url, params={'year': str(year)})

    # Have to use html5lib, because default HTML parser wasn't working for this site
    soup = BeautifulSoup(response.text, 'html5lib')
    return soup.find('div', class_='datadiv')


def main(project_path, page_args=PAGES):
    today = datetime.now()
    # AFL Grand Final seems to be on Saturday of last partial week of September
    # (meaning it sometimes falls within first few days of October),
    # but let's keep the math simple and say a given season goes through October
    season_year = today.year if today.month > 10 else today.year - 1
    data = []
    for page in PAGES:
        page_url = f'{DOMAIN}{PATH}{page}'
        # Data for each season are on different pages, so looping back through years
        # until no data is returned.
        # NOTE: This can't be refactored, because we need to be able to break the loop
        # once a blank page is returned.
        for year in reversed(range(season_year + 1)):
            data_div = fetch_page_data(page_url, year)

            if data_div is None:
                break

            if page == 'ft_match_list':
                data.append(fixture_data(data_div, year))
            elif page == 'afl_betting':
                data.append(betting_data(data_div, year))

        if len(data) > 0:
            max_length = len(max(data, key=len))
            # Add null cells, so all rows are same length for Pandas dataframe
            padded_data = [list(row) + [None] * (max_length - len(row)) for row in data]

            with open(os.path.join(project_path, f'data/raw/{page}.csv'), 'w') as csv_file:
                wr = csv.writer(csv_file)
                wr.writerows(padded_data)


if __name__ == '__main__':
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if project_path not in sys.path:
        sys.path.append(project_path)

    page_args = list(sys.argv[1:]) if len(sys.argv) > 1 else None

    main(project_path, page_args=page_args)
