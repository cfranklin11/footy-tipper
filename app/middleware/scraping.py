import os
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import dateutil
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, BettingOdds, Team


TEAM_TRANSLATION = {
    'Tigers': 'Richmond',
    'Blues': 'Carlton',
    'Demons': 'Melbourne',
    'Giants': 'GWS',
    'Suns': 'Gold Coast',
    'Bombers': 'Essendon',
    'Swans': 'Sydney',
    'Magpies': 'Collingwood',
    'Kangaroos': 'North Melbourne',
    'Crows': 'Adelaide',
    'Bulldogs': 'Western Bulldogs',
    'Dockers': 'Fremantle',
    'Power': 'Port Adelaide',
    'Saints': 'St Kilda',
    'Eagles': 'West Coast',
    'Lions': 'Brisbane',
    'Cats': 'Geelong',
    'Hawks': 'Hawthorn'
}
VENUE_TRANSLATION = {
    'AAMI': 'AAMI Stadium',
    'ANZ': 'ANZ Stadium',
    'Adelaide': 'Adelaide Oval',
    'Aurora': 'Aurora Stadium',
    'Blacktown': 'Blacktown International',
    'Blundstone': 'Blundstone Arena',
    "Cazaly's": "Cazaly's Stadium",
    'Domain': 'Domain Stadium',
    'Etihad': 'Etihad Stadium',
    'GMHBA': 'GMHBA Stadium',
    'Gabba': 'Gabba',
    'Jiangwan': 'Adelaide Arena at Jiangwan Stadium',
    'MCG': 'MCG',
    'Mars': 'Mars Stadium',
    'Metricon': 'Metricon Stadium',
    'SCG': 'SCG',
    'Spotless': 'Spotless Stadium',
    'StarTrack': 'Manuka Oval',
    'TIO': 'TIO Stadium',
    'UTAS': 'UTAS Stadium',
    'Westpac': 'Westpac Stadium',
    'TIO Traegar Park': 'TIO Stadium'
}

# Extract both regular season rounds and finals rounds
# NOTE: Regex uses \s+, because I encountered a case where 'Qualifying' and 'Final'
# had two spaces instead of one
ROUND_REGEX = re.compile('(round\s+\d\d?|.*final.*)', flags=re.I)
ROUND_NUM_REGEX = re.compile('round\s+(\d\d?)', flags=re.I)
MATCH_COL_NAMES = ['year', 'date', 'home_team', 'away_team', 'venue', 'result']
MATCH_COL_INDICES = [0, 1, 2, 4, 5, 7]
BETTING_COL_NAMES = ['year', 'date', 'venue', 'team', 'score', 'margin', 'win_odds',
                     'win_paid', 'point_spread']
BETTING_COL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

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


class PageDataCleaner():
    def __init__(self, page_data):
        self.page_data = page_data

    def data(self):
        data_dfs = {}
        for page in self.page_data:
            if page['name'] == 'ft_match_list':
                data_dfs['match'] = self.__clean_match_data(page['data'])
            elif page['name'] == 'afl_betting':
                data_dfs['betting_odds'] = self.__clean_betting_data(page['data'])
            else:
                raise Exception(f'Unknown page: {page}')

        return data_dfs

    def __clean_match_data(self, data):
            # Ignore useless columns that are result of BeautifulSoup table parsing
        df = pd.DataFrame(data).iloc[:, MATCH_COL_INDICES]
        df.columns = MATCH_COL_NAMES

        df = df.assign(season_round=self.__get_season_round)
        df.loc[:, 'home_team'] = df['home_team'].apply(
            lambda x: np.nan if x in ['BYE', 'MATCH CANCELLED'] else x
        )
        # Round label just appears at top of round in table,
        # so forward fill to apply it to all relevant matches
        df.loc[:, 'season_round'].fillna(method='ffill', inplace=True)

        # We need to do this in two steps because there's at least one case of the website's data table
        # being poorly organised, leading to specious duplicates.
        # So, we fill the season_round column before dropping duplicates, then we assign home_score
        # and away_score to avoid NaNs that will raise errors.
        df = df.drop_duplicates(
            # Check all columns except for round #, because round # would make all rows
            # unique
            subset=df.columns.values[:-1],
            # Duplicate rows are from table labels/headers that are not useful
            keep=False
        ).assign(
            # Result column has format: 'home_score-away_score'
            home_score=self.__get_home_score,
            away_score=self.__get_away_score
        ).drop(
            'result', axis=1
            # The only rows with NaNs are the round label rows that we no longer need
        ).dropna(
        ).reset_index(
            drop=True
        ).assign(
            # Save date parsing till the end to avoid ValueErrors
            date=self.__create_full_match_date
        ).drop(
            ['year'], axis=1
        )

        df.loc[:, 'venue'] = df['venue'].apply(
            lambda x: VENUE_TRANSLATION[x] if x in VENUE_TRANSLATION.keys() else x
        )

        return df

    def __clean_betting_data(self, data):
        # Ignore useless columns that are result of BeautifulSoup table parsing
        df = pd.DataFrame(data).iloc[:, BETTING_COL_INDICES]
        df.columns = BETTING_COL_NAMES
        df = df.assign(season_round=self.__get_season_round)

        df.loc[:, 'team'] = df['team'].apply(
            lambda x: TEAM_TRANSLATION[x] if x in TEAM_TRANSLATION.keys() else x
        )
        # Round label just appears at top of round in table,
        # so forward fill to apply it to all relevant matches
        df.loc[:, 'season_round'].fillna(method='ffill', inplace=True)

        # Betting data table uses colspan="2" for date columns, making even dates blank
        df.loc[:, 'date'].fillna(method='ffill', inplace=True)
        df.loc[:, 'venue'] = df['venue'].fillna(
            method='ffill'
        ).apply(
            lambda x: VENUE_TRANSLATION[x] if x in VENUE_TRANSLATION.keys() else x
        )

        # We need to do this in two steps because there's at least one case of the website's data table
        # being poorly organised, leading to specious duplicates.
        # So, we fill the season_round column before dropping duplicates, then we assign home_score
        # and away_score to avoid NaNs that will raise errors.
        df = df.drop_duplicates(
            # Check all columns that have column labels from the html tables
            subset=df.columns.values[3:9],
            # Duplicate rows are from table labels/headers that are not useful, so remove all
            keep=False
        ).dropna(
        ).reset_index(
            drop=True,
        ).drop(
            ['year', 'score', 'win_paid', 'margin'], axis=1
        )

        # Save date parsing till the end to avoid ValueErrors
        df.loc[:, 'date'] = df['date'].apply(dateutil.parser.parse)

        return df

    def __get_season_round(self, df):
        return df['date'].str.extract(ROUND_REGEX, expand=True)

    def __get_home_score(self, df):
        results = df['result'].str.split('-')
        return results.apply(lambda x: float(x[0]) if type(x) == list and len(x) == 2 else 0)

    def __get_away_score(self, df):
        results = df['result'].str.split('-')
        return results.apply(lambda x: float(x[1]) if type(x) == list and len(x) == 2 else 0)

    def __create_full_match_date(self, df):
        # year is a float, so we have to convert it to int, then str to concatenate
        # with the date string (date parser doesn't recognize years as floats)
        # We convert datetime to date, so match dates can be paired with betting data,
        # which doesn't include match times.
        return (df['date'] + df['year'].astype(int).astype(str)).apply(
            dateutil.parser.parse
        ).apply(
            datetime.date
        )


class DataSaver():
    def __init__(self, data):
        self.data = data

    def save_data(self):
        engine = create_engine(app.config['DATABASE_URL'])
        Session = sessionmaker(bind=engine)
        session = Session()

        teams = session.query(Team).all()

        if 'match' in self.data.keys():
            self.__save_match_data(session, teams)
        if 'betting_odds' in self.data.keys():
            self.__save_betting_data(session, teams)

        try:
            session.commit()
        except:
            print('Something went wrong, rolling back')
            session.rollback()
            raise
        finally:
            print('Closing session...')
            session.close()

    def __save_match_data(self, session, teams):
        match_df = self.data['match'].assign(
            round_number=lambda x: x['season_round'].str.extract(ROUND_NUM_REGEX, expand=True).astype(int)
        )

        played_match_dates = match_df[(match_df['home_score'] != 0) & (match_df['away_score'] != 0)]['date']

        # If matches have been played this year, get latest played round number
        if len(played_match_dates) > 0:
            last_date_played = max(played_match_dates)
            last_round_played = match_df[
                match_df['date'] == last_date_played
            ]['round_number'].drop_duplicates().values

            if len(last_round_played > 1):
                raise(Exception(
                    f'More than one season found on date {last_date_played}: {last_round_played}'
                ))

            last_round_number_played = int(last_round_played[0])
        else:
            last_round_number_played = 0

        db_matches = session.query(
            Match
        ).filter(
            Match.date > datetime(datetime.now().year, 1, 1)
        ).all()
        match_records = match_df.to_dict('records')

        for match_record in match_records:
            try:
                db_match = next((
                    match for match in db_matches if (
                        # Have to convert DF date to datetime for equality comparison with DB datetime
                        match.date == datetime.combine(match_record['date'], datetime.min.time()) and
                        match.venue == match_record['venue']
                    )
                ))
            except StopIteration:
                db_match = None

            match_dict = {
                'date': match_record['date'],
                'season_round': match_record['season_round'],
                'venue': match_record['venue'],
                'home_score': match_record['home_score'],
                'away_score': match_record['away_score'],
                'home_team': next((team for team in teams if team.name == match_record['home_team'])),
                'away_team': next((team for team in teams if team.name == match_record['away_team']))
            }

            if db_match is None:
                # Skip to next if it's a next week's round or later
                if match_record['round_number'] > last_round_number_played + 1:
                    continue

                # Raise exception if it's this week's round, but the score's aren't 0
                if (match_record['round_number'] == last_round_number_played + 1 and
                   (match_record['home_score'] != 0 or match_record['away_score'] != 0)):
                    raise(Exception('Expected scores from matches from this round to be 0. ' +
                                    f'Instead got {match_record}'))

                # Update any missing data from past rounds from this season and
                # save this week's matches for predicting results
                session.add(Match(**match_dict))
            else:
                if (db_match.home_score == match_record['home_score'] and
                   db_match.away_score == match_record['away_score']):
                    continue

                if db_match.home_score > 0 or db_match.away_score > 0:
                    raise(Exception(
                        'Expected older match data in DB to be the same as match data ' +
                        f'scraped from webpages. Instead got {db_match} from DB ' +
                        f'and {match_record} from webpage.'
                    ))

                # Update last week's match data with scores
                db_match.home_score == match_record['home_score']
                db_match.away_score == match_record['away_score']

    def __save_betting_data(self, session, teams):
        db_betting_odds = session.query(
            BettingOdds
        ).filter(
            BettingOdds.date > datetime(datetime.now().year, 1, 1)
        ).all()
        betting_df = self.data['betting_odds']
        betting_records = betting_df.to_dict('records')
        db_matches = session.query(
            Match
        ).filter(
            BettingOdds.date > datetime(datetime.now().year, 1, 1)
        ).all()

        for betting_record in betting_records:
            # Skip to next if there's no betting data
            if betting_record['win_odds'] == 0 or betting_record['point_spread'] == 0:
                continue

            try:
                db_betting = next((
                    betting for betting in db_betting_odds if (
                        # Have to convert DF date to datetime for equality comparison with DB datetime
                        betting.home_match.date == datetime.combine(
                            betting_record['date'], datetime.min.time()
                        ) and
                        betting.home_match.venue == betting_record['venue'] and
                        betting.team.name == betting_record['team']
                    )
                ))
            except StopIteration:
                db_betting = None

            if db_betting is None:
                try:
                    betting_match = next((
                        match for match in db_matches if (
                            match.date == datetime.combine(betting_record['date'], datetime.min.time()) and
                            match.venue == betting_record['venue']
                        )
                    ))
                except StopIteration:
                    # If the betting record has data but no associated match, raise an exception
                    raise(Exception(f'No match found for betting data: {betting_record}'))

                betting_team = next((
                    team for team in teams if team.name == betting_record['team']
                ))
                betting_dict = {
                    'win_odds': betting_record['win_odds'],
                    'point_spread': betting_record['point_spread'],
                    'team': betting_team
                }

                if betting_team.id == betting_match.home_team_id:
                    betting_dict['home_match'] = betting_match
                elif betting_team.id == betting_match.away_team_id:
                    betting_dict['away_match'] = betting_match
                else:
                    raise(Exception(
                        f'Betting data {betting_record} does not match any existing ' +
                        'team/match combinations'
                    ))

                session.add(BettingOdds(**betting_dict))
            else:
                db_betting.win_odds = betting_record['win_odds']
                db_betting.point_spread = betting_record['point_spread']
