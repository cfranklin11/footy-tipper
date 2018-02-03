import os
import pandas as pd
import numpy as np
import dateutil
from datetime import datetime
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
BETTING_COL_NAMES = ['year', 'date', 'venue', 'team', 'score', 'margin', 'win_odds',
                     'win_paid', 'point_spread']
BETTING_COL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def get_season_round(df):
    return df['date'].str.extract(ROUND_REGEX, expand=True)


def get_home_score(df):
    results = df['result'].str.split('-')
    return results.apply(lambda x: float(x[0]) if type(x) == list and len(x) == 2 else None)


def get_away_score(df):
    results = df['result'].str.split('-')
    return results.apply(lambda x: float(x[1]) if type(x) == list and len(x) == 2 else None)


def create_full_match_date(df):
    # year is a float, so we have to convert it to int, then str to concatenate
    # with the date string (date parser doesn't recognize years as floats)
    # We convert datetime to date, so match dates can be paired with betting data,
    # which doesn't include match times.
    return (df['date'] + df['year'].astype(int).astype(str)).apply(
        dateutil.parser.parse
    ).apply(
        datetime.date
    )


def clean_match_data(data):
        # Ignore useless columns that are result of BeautifulSoup table parsing
    df = pd.DataFrame(data).iloc[:, MATCH_COL_INDICES]
    df.columns = MATCH_COL_NAMES

    df = df.assign(season_round=get_season_round)
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
        # Save date parsing till the end to avoid ValueErrors
        full_date=create_full_match_date
    ).drop(
        ['year', 'date'], axis=1
    )
    df.loc[:, 'venue'] = df['venue'].apply(
        lambda x: VENUE_TRANSLATION[x] if x in VENUE_TRANSLATION.keys() else x
    )

    return df


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
    ).assign(
        # Save date parsing till the end to avoid ValueErrors
        full_date=lambda x: x['date'].apply(dateutil.parser.parse)
    ).drop(
        ['year', 'date', 'score', 'win_paid', 'margin'], axis=1
    )

    return df


def main(page, data, csv=False):
    if page == 'ft_match_list':
        df = clean_match_data(data)
    elif page == 'afl_betting':
        df = clean_betting_data(data)
    else:
        raise Exception(f'Unknown page: {page}')

    if csv:
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        data_directory = os.path.join(project_path, 'data')

        if not os.path.isdir(data_directory):
            os.makedirs(data_directory)

        df.to_csv(os.path.join(data_directory, f'{page}.csv'), index=False)

    return df
