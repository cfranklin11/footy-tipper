import pandas as pd
import numpy as np
import dateutil
import datetime
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
    'Hawks': 'Hawthorn'
}
ROUND_REGEX = re.compile('(Round\s+\d\d?|(?:\w|\d)+\s+Final)')
ROUND_NUMBER_REGEX = re.compile('(.*[Ff]inal.*|\d\d?)')
MATCH_COL_NAMES = ['year', 'date', 'home_team', 'away_team', 'venue', 'result']
MATCH_COL_INDICES = [0, 1, 2, 4, 5, 7]
BETTING_COL_NAMES = ['year', 'date', 'venue', 'team', 'score', 'margin', 'win_odds',
                     'win_paid', 'point_spread']
BETTING_COL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def create_match_date(idx, date):
    # Add a few seconds to match times to guarantee uniqueness,
    # because some occur on the same day at the same time
    delta = idx % 60
    return (date + datetime.timedelta(0, delta))


def get_season_round(df):
    # Extract both regular season rounds and finals rounds
    # NOTE: Regex uses \s+, because I encountered a case where 'Qualifying' and 'Final'
    # had two spaces instead of one
    return df['date'].str.extract(ROUND_REGEX, expand=True)


def get_home_score(df):
    results = df['result'].str.split('-')
    return results.apply(lambda x: float(x[0]) if type(x) == list else None)


def get_away_score(df):
    results = df['result'].str.split('-')
    return results.apply(lambda x: float(x[1]) if type(x) == list else None)


def create_round_number(df):
    return df['season_round'].str.extract(ROUND_NUMBER_REGEX, expand=False)


def create_full_match_date(df):
    # year is a float, so we have to convert it to int, then str to concatenate
    # with the date string (date parser doesn't recognize years as floats)
    full_date = (df['date'] + df['year'].astype(int).astype(str)).apply(dateutil.parser.parse)

    # Add a few seconds to match times to guarantee uniqueness,
    # because some occur on the same day at the same time
    return [create_match_date(idx, match_date) for idx, match_date in enumerate(full_date)]


def clean_match_data(data):
        # Ignore useless columns that are result of BeautifulSoup table parsing
    df = pd.DataFrame(data).iloc[:, MATCH_COL_INDICES]
    df.columns = MATCH_COL_NAMES

    df = df.assign(season_round=get_season_round, inplace=True)
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
        home_score=get_home_score,
        away_score=get_away_score
    ).drop(
        'result', axis=1
        # The only rows with NaNs are the round label rows that we no longer need
    ).dropna(
    ).reset_index(
        drop=True
    ).assign(
        round_number=create_round_number,
        # Save date parsing till the end to avoid ValueErrors
        full_date=create_full_match_date
    )

    df.set_index('full_date', inplace=True)
    df.to_csv('data/interim/match_data.csv')


def create_full_betting_date(df):
    full_date = df['date'].apply(dateutil.parser.parse)
    # Add a few seconds to match times to guarantee uniqueness,
    # because some occur on the same day at the same time
    return [create_match_date(idx, match_date) for idx, match_date in enumerate(full_date)]


def clean_betting_data(data):
    # Ignore useless columns that are result of BeautifulSoup table parsing
    df = pd.DataFrame(data).iloc[:, BETTING_COL_INDICES]
    df.columns = BETTING_COL_NAMES
    df = df.assign(season_round=get_season_round)

    df.loc[:, 'team'] = df['team'].apply(
        lambda x: TEAM_TRANSLATION[x] if x in TEAM_TRANSLATION.keys() else x
    )
    # Round label just appears at top of round in table,
    # so forward fill to apply it to all relevant matches
    df.loc[:, 'season_round'].fillna(method='ffill', inplace=True)

    # Betting data table uses colspan="2" for date columns, making even dates blank
    df.loc[:, 'date'].fillna(method='ffill', inplace=True)
    df.loc[:, 'venue'].fillna(method='ffill', inplace=True)

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
    ).assign(
        round_number=create_round_number,
        # Save date parsing till the end to avoid ValueErrors
        full_date=create_full_betting_date
    )

    df.set_index('full_date', inplace=True)
    df.to_csv('data/interim/betting_data.csv')


def main(data_list):
    for data in data_list:
        if data['name'] == 'ft_match_list':
            clean_match_data(data['data'])
        elif data['name'] == 'afl_betting.csv':
            clean_betting_data(data['data'])
