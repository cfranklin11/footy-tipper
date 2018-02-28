from datetime import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup

DOMAIN = 'https://www.footywire.com'
PATH = '/afl/footy/'
PAGES = [
    # Default year for data is most recent season. Append '?year=####' to get different years
    'ft_match_list',
    'afl_betting'
]


class PageScraper():
    def __init__(self):
        self.year = datetime.now().year

    def data(self):
        data = self.__scrape_pages()

        return data

    def __scrape_pages(self):
        page_data = []

        for page in PAGES:
            page_url = f'{DOMAIN}{PATH}{page}'

            data_div = self.__fetch_page_data(page_url)
            if data_div is None:
                raise(Exception(f"Couldn't find div with class 'datadiv' on {page}"))

            if page == 'ft_match_list':
                data = self.__fixture_data(data_div)
            # TODO: 01-02-2018: The afl_betting page for 2018 doesn't exist yet
            # (unlike ft_match_list, which lists future matches).
            # Hopefully they'll update it as we get closer to the start of the season
            # and betting odds get set.
            elif page == 'afl_betting':
                data = self.__betting_data(data_div)

            if len(data) > 0:
                max_length = len(max(data, key=len))
                # Add null cells, so all rows are same length for Pandas dataframe
                padded_data = [list(row) + [None] * (max_length - len(row)) for row in data]

                page_data.append({'name': page, 'data': padded_data})

        return page_data

    def __fetch_page_data(self, page_url):
        response = requests.get(page_url, params={'year': str(self.year)})
        text = response.text
        # Have to use html5lib, because default HTML parser wasn't working for this site
        soup = BeautifulSoup(text, 'html5lib')

        return soup.find('div', class_='datadiv')

    def __fixture_data(self, data_div):
        data_table = data_div.find('table')

        if data_table is None:
            raise(Exception("Couldn't find data table for ft_match_list"))

        return [self.__get_fixture_row(tr) for tr in data_table.find_all('tr')]

    def __get_fixture_row(self, tr):
        table_row = list(tr.stripped_strings)

        if len(table_row) == 0:
            return []

        return [self.year] + table_row

    def __betting_data(self, data_div):
        # afl_betting page nests the data table inside of an outer table
        data_table = data_div.find('table').find('table')

        if data_table is None:
            return None

        return [self.__get_betting_row(tr) for tr in data_table.find_all('tr')]

    def __get_betting_row(self, tr):
            table_row = list(tr.stripped_strings)

            if len(table_row) == 0:
                return []

            # First two columns in data rows have colspan="2", so empty cells need to be prepended
            # to every-other data row. There doesn't seem to be a good way of identifying these rows
            # apart from their length: 11 cells means the date is in the row, 9 means there's no date
            if len(table_row) == 9:
                return [self.year] + ([np.nan] * 2) + table_row

            return [self.year] + table_row
