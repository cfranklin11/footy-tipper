from datetime import datetime
import re
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
LADBROKES_URL = 'https://www.ladbrokes.com.au/sports/australian-rules/'
DATA_CLASS_REGEX = re.compile('sports\-(?:sub)?header|market\-row')


class PageScraper():
    # NOTE: As of 17-03-2018, Footywire doesn't seem to be updating
    # their betting data, so I'm scraping ladbrokes.com.au for weekly
    # betting data, but leaving existing code in case the situation changes
    def __init__(self, footywire=False):
        self.footywire = footywire
        self._year = datetime.now().year

    def data(self):
        data = self.__scrape_pages()

        return data

    def __scrape_pages(self):
        page_data = {}

        for page in PAGES:
            if page == 'ft_match_list' or self.footywire:
                page_url = '{}{}{}'.format(DOMAIN, PATH, page)
                soup = self.__fetch_page_data(page_url, params={'year': str(self._year)})
                data_div = soup.find('div', class_='datadiv')
            else:
                soup = self.__fetch_page_data(LADBROKES_URL)
                data_div = soup.find('div', class_='float-left seasonal-pricing')

            if data_div is None:
                raise(Exception("Couldn't find div with class 'datadiv' on {}".format(page)))

            if page == 'ft_match_list':
                data = self.__fixture_data(data_div)
            elif page == 'afl_betting':
                data = self.__betting_data(data_div)

            if len(data) > 0:
                max_length = len(max(data, key=len))
                # Add null cells, so all rows are same length for Pandas dataframe
                padded_data = [list(row) + [None] * (max_length - len(row)) for row in data]

                page_data[page] = padded_data

        return page_data

    def __fetch_page_data(self, page_url, params=None):
        response = requests.get(page_url, params=params)
        text = response.text
        # Have to use html5lib, because default HTML parser wasn't working for this site
        return BeautifulSoup(text, 'html5lib')

    def __fixture_data(self, data_div):
        data_table = data_div.find('table')

        if data_table is None:
            raise(Exception("Couldn't find data table for ft_match_list"))

        return [self.__get_fixture_row(tr) for tr in data_table.find_all('tr')]

    def __get_fixture_row(self, tr):
        table_row = list(tr.stripped_strings)

        if len(table_row) == 0:
            return []

        return [self._year] + table_row

    def __betting_data(self, data_div):
        if self.footywire:
            # afl_betting page nests the data table inside of an outer table
            data_table = data_div.find('table').find('table')

            if data_table is None:
                return None

            data = data_table.find_all('tr')
        else:
            data = data_div.find_all('div', attrs={'class': DATA_CLASS_REGEX})

        return [self.__get_betting_row(data_row) for data_row in data]

    def __get_betting_row(self, data_row):
            row_strings = list(data_row.stripped_strings)

            if self.footywire:
                if len(row_strings) == 0:
                    return []

                # First two columns in data rows have colspan="2", so empty cells need to be prepended
                # to every-other data row. There doesn't seem to be a good way of identifying these rows
                # apart from their length: 11 cells means the date is in the row, 9 means there's no date
                if len(row_strings) == 9:
                    return [self._year] + ([np.nan] * 2) + row_strings

                return [self._year] + row_strings

            return row_strings
