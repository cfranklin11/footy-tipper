from datetime import datetime
import numpy as np
import pandas as pd
import dateutil
import re


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
    'Hawks': 'Hawthorn',
    'Adelaide Crows': 'Adelaide',
    'Brisbane Lions': 'Brisbane',
    'Gold Coast Suns': 'Gold Coast',
    'GWS Giants': 'GWS',
    'Geelong Cats': 'Geelong',
    'West Coast Eagles': 'West Coast',
    'Sydney Swans': 'Sydney'
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
MATCH_COL_NAMES = ['year', 'date', 'home_team', 'away_team', 'venue', 'result']
MATCH_COL_INDICES = [0, 1, 2, 4, 5, 7]
FW_BETTING_COL_NAMES = ['year', 'date', 'venue', 'team', 'score', 'margin', 'win_odds',
                        'win_paid', 'point_spread']
FW_BETTING_COL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
LB_BETTING_COL_INDICES = [0, 1, 3]
LB_BETTING_COL_NAMES = {0: 'team', 1: 'win_odds', 3: 'line_odds'}


class DataCleaner():
    def __init__(self, page_data, footywire=False):
        self.page_data = page_data
        self.footywire = footywire

    def data(self):
        data_dfs = {}

        if 'ft_match_list' in self.page_data.keys():
            data_dfs['match'] = self.__clean_match_data(self.page_data['ft_match_list'])
        if 'afl_betting' in self.page_data.keys():
            # Data cleaning for the two different sources of betting data is too
            # different to put in the same method
            if self.footywire:
                data_dfs['betting_odds'] = self.__clean_fw_betting_data(
                    self.page_data['afl_betting']
                )
            else:
                data_dfs['betting_odds'] = self.__clean_lb_betting_data(
                    self.page_data['afl_betting'], data_dfs['match']
                )

        return data_dfs

    def __clean_match_data(self, data):
        # Ignore useless columns that are result of BeautifulSoup table parsing
        df = pd.DataFrame(data).iloc[:, MATCH_COL_INDICES]
        df.columns = MATCH_COL_NAMES

        df = df.assign(season_round=self.__get_season_round('date'))
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

    def __clean_fw_betting_data(self, data):
        # Ignore useless columns that are result of BeautifulSoup table parsing
        df = pd.DataFrame(data).iloc[:, FW_BETTING_COL_INDICES]
        df.columns = FW_BETTING_COL_NAMES
        df = df.assign(season_round=self.__get_season_round('date'))

        df.loc[:, 'team'] = df['team'].apply(self.__translate_teams)
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

    def __clean_lb_betting_data(self, data, match_df):
        df = (pd.DataFrame(data)
                .iloc[:, LB_BETTING_COL_INDICES]
                .rename(LB_BETTING_COL_NAMES, axis=1)
              # Season round and date data is in first column ('team'), so we extract
              # them, then drop NaNs
                .assign(season_round=self.__get_season_round('team'),
                        date=lambda x: pd.to_datetime(x['team'], errors='coerce', dayfirst=True),
                        point_spread=lambda x: (
                            x['line_odds'].str.split(' @ ', expand=True)[0].astype(float)))
                .drop('line_odds', axis=1)
              )

        df.loc[:, ['season_round', 'date']] = df[['season_round', 'date']].ffill()
        df.dropna(inplace=True)
        df.loc[:, 'date'] = df['date'].apply(datetime.date)
        df.loc[:, 'team'] = df['team'].apply(self.__translate_teams)

        venue_df = pd.DataFrame({
            'team': pd.concat([match_df['home_team'], match_df['away_team']]),
            'venue': pd.concat([match_df['venue'], match_df['venue']]),
            'date': pd.concat([match_df['date'], match_df['date']])
        })

        return df.merge(venue_df, on=['team', 'date'], how='left')

    def __get_season_round(self, column):
        return lambda x: x[column].str.extract(ROUND_REGEX, expand=True)

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

    def __translate_teams(self, x):
        return TEAM_TRANSLATION[x] if x in TEAM_TRANSLATION.keys() else x
