import os
import sys
import re
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, BettingOdds, Team

ROUND_NUM_REGEX = re.compile('round\s+(\d\d?)', flags=re.I)


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
