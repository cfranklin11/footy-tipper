import os
import sys
import re
from datetime import datetime
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker, aliased

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, BettingOdds, Team


DIGITS = re.compile('(?:round\s+)?(\d+)$', flags=re.I)
QUALIFYING = re.compile('qualifying', flags=re.I)
ELIMINATION = re.compile('elimination', flags=re.I)
SEMI = re.compile('semi', flags=re.I)
PRELIMINARY = re.compile('preliminary', flags=re.I)
GRAND = re.compile('grand', flags=re.I)


class DataSaver():
    def __init__(self, data):
        self.data = data

    def save_data(self):
        engine = create_engine(app.config['DATABASE_URL'])
        Session = sessionmaker(bind=engine)
        session = Session()

        teams = session.query(Team).all()

        # We have to save match first, because wheter we save betting odds is dependent
        # on having a corresponding match
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
        match_df = self.data['match'].assign(round_number=self.__get_finals_round_numbers)
        self._last_round_number_played = self.__last_round_number_played(match_df)

        # Limit query to matches from this year since this class is meant to update
        # current season matches only
        db_matches = session.query(
            Match
        ).filter(
            Match.date > datetime(datetime.now().year, 1, 1)
        ).all()
        scraped_matches = match_df.to_dict('records')

        # Get list of tuples of duplicate scraped and db matches
        duplicate_matches = [self.__duplicate_matches(scraped_match, db_matches)
                             for scraped_match in scraped_matches]
        duplicate_matches = [duplicate for duplicate in duplicate_matches if duplicate is not None]

        if len(duplicate_matches) > 0:
            # Convert list of tuples into two lists
            db_duplicates, scraped_duplicates = zip(*duplicate_matches)
        else:
            db_duplicates, scraped_duplicates = (), ()

        matches_to_save = [
            self.__match_to_save(scraped_match, teams)
            for scraped_match in scraped_matches
            if (
                scraped_match not in scraped_duplicates and
                scraped_match['round_number'] <= self._last_round_number_played + 1
            )
        ]
        session.add_all(matches_to_save)

        self.__update_db_match_scores(db_duplicates, scraped_duplicates)

    def __get_finals_round_numbers(self, df):
        return df['season_round'].apply(self.__get_finals_round_number)

    def __get_finals_round_number(self, season_round):
        digits = DIGITS.search(season_round)
        if digits is not None:
            return int(digits.group(1))
        if QUALIFYING.search(season_round) is not None:
            return 25
        if ELIMINATION.search(season_round) is not None:
            return 25
        if SEMI.search(season_round) is not None:
            return 26
        if PRELIMINARY.search(season_round) is not None:
            return 27
        if GRAND.search(season_round) is not None:
            return 28

        raise Exception("Round label doesn't match any known patterns")

    def __last_round_number_played(self, df):
        played_match_dates = df[(df['home_score'] != 0) & (df['away_score'] != 0)]['date']

        # If matches have been played this year, get latest played round number
        if len(played_match_dates) > 0:
            last_date_played = max(played_match_dates)
            last_round_played = df[
                df['date'] == last_date_played
            ]['round_number'].drop_duplicates().values

            if len(last_round_played) > 1:
                raise(Exception(
                    'More than one season found on date {}: {}'.format(last_date_played, last_round_played)
                ))

            return int(last_round_played[0])
        else:
            return 0

    def __duplicate_matches(self, scraped_match, db_matches):
        duplicate_db_matches = [
            db_match for db_match in db_matches if (
                # Have to convert DF date to datetime for equality comparison with DB datetime
                db_match.date == datetime.combine(scraped_match['date'], datetime.min.time()) and
                db_match.venue == scraped_match['venue']
            )
        ]

        if len(duplicate_db_matches) == 0:
            return

        if len(duplicate_db_matches) > 1:
            raise(Exception('Unexpected duplication of match date and venue in DB matches. ' +
                            'Expected 0 or 1, but found {} records.'.format(len(duplicate_db_matches))))

        return (duplicate_db_matches[0], scraped_match)

    def __match_to_save(self, scraped_match, teams):
        # Raise exception if it's this week's round, but the score's aren't 0
        if (scraped_match['round_number'] == self._last_round_number_played + 1 and
           (scraped_match['home_score'] != 0 or scraped_match['away_score'] != 0)):
            raise(Exception('Expected scores from matches from this round to be 0. ' +
                            'Instead got {}'.format(scraped_match)))

        match_dict = {
            'date': scraped_match['date'],
            'season_round': scraped_match['season_round'],
            'venue': scraped_match['venue'],
            'home_score': scraped_match['home_score'],
            'away_score': scraped_match['away_score'],
            'home_team': next((team for team in teams if team.name == scraped_match['home_team'])),
            'away_team': next((team for team in teams if team.name == scraped_match['away_team']))
        }

        return Match(**match_dict)

    def __update_db_match_scores(self, db_duplicates, scraped_duplicates):
        for idx, db_match in enumerate(db_duplicates):
            scraped_match = scraped_duplicates[idx]

            if (db_match.home_score == scraped_match['home_score'] and
               db_match.away_score == scraped_match['away_score']):
                continue

            if db_match.home_score > 0 or db_match.away_score > 0:
                raise(Exception(
                    'Expected older match data in DB to be the same as match data ' +
                    'scraped from webpages. Instead got {} from DB '.format(db_match) +
                    'and {} from webpage.'.format(scraped_match)
                ))

            # Update last week's match data with scores
            db_match.home_score == scraped_match['home_score']
            db_match.away_score == scraped_match['away_score']

    def __save_betting_data(self, session, teams):
        current_year = datetime.now().year
        HomeMatch = aliased(Match)
        AwayMatch = aliased(Match)
        # This class is for updating data during the season, so only matches/betting odds
        # from this year are relevant
        db_betting_odds = (session.query(BettingOdds)
                                  .outerjoin(HomeMatch, BettingOdds.home_match)
                                  .outerjoin(AwayMatch, BettingOdds.away_match)
                                  .filter(or_(HomeMatch.date > datetime(current_year, 1, 1),
                                              AwayMatch.date > datetime(current_year, 1, 1)))
                                  .all())
        db_matches = (session.query(Match)
                             .filter(Match.date > datetime(current_year, 1, 1))
                             .all())
        scraped_betting_odds = self.data['betting_odds'].to_dict('records')

        # Get list of tuples of duplicate scraped and db matches
        duplicate_betting_odds = [
            self.__duplicate_betting_odds(scraped_betting_odd, db_betting_odds)
            for scraped_betting_odd in scraped_betting_odds
        ]
        duplicate_betting_odds = [
            duplicate for duplicate in duplicate_betting_odds if duplicate is not None
        ]

        if len(duplicate_betting_odds) > 0:
            # Convert list of tuples into two tuples containing first & second elements
            # of each listed tuple
            db_duplicates, scraped_duplicates = zip(
                *[duplicate for duplicate in duplicate_betting_odds if duplicate is not None]
            )
        else:
            db_duplicates, scraped_duplicates = (), ()

        betting_odds_to_save = [
            self.__betting_odds_to_save(scraped_betting_odd, db_matches, teams)
            for scraped_betting_odd in scraped_betting_odds
            if scraped_betting_odd not in scraped_duplicates
        ]
        betting_odds_to_save = [
            betting_odd_to_save for betting_odd_to_save in betting_odds_to_save
            if betting_odd_to_save is not None
        ]
        session.add_all(betting_odds_to_save)

        self.__update_db_betting_odds(db_duplicates, scraped_duplicates)

    def __duplicate_betting_odds(self, scraped_betting_odd, db_betting_odds):
        betting_odds_date = datetime.combine(scraped_betting_odd['date'], datetime.min.time())
        duplicate_db_betting_odds = [
            db_betting_odd for db_betting_odd in db_betting_odds
            # Have to convert DF date to datetime for equality comparison with DB datetime
            if (db_betting_odd.date() == betting_odds_date and
                db_betting_odd.venue() == scraped_betting_odd['venue'] and
                db_betting_odd.team.name == scraped_betting_odd['team'])
        ]

        # Skip to next if there's no betting data or no duplicates
        if (
            scraped_betting_odd['win_odds'] == 0 or
            scraped_betting_odd['point_spread'] == 0 or
            len(duplicate_db_betting_odds) == 0
        ):
            return

        if len(duplicate_db_betting_odds) > 1:
            raise(Exception('Unexpected duplication of betting date, venue, and team in DB betting odds. ' +
                            'Expected 0 or 1, but found {} records.'.format(len(duplicate_db_betting_odds))))

        return (duplicate_db_betting_odds[0], scraped_betting_odd)

    def __betting_odds_to_save(self, scraped_betting_odd, db_matches, teams):
        betting_match = [
            match for match in db_matches
            if (match.date == datetime.combine(scraped_betting_odd['date'], datetime.min.time()) and
                match.venue == scraped_betting_odd['venue'])
        ]

        if len(betting_match) == 0:
            return

        betting_team = next((
            team for team in teams if team.name == scraped_betting_odd['team']
        ))
        betting_dict = {
            'win_odds': scraped_betting_odd['win_odds'],
            'point_spread': scraped_betting_odd['point_spread'],
            'team': betting_team
        }

        if betting_team.id == betting_match[0].home_team_id:
            betting_dict['home_match'] = betting_match[0]
        elif betting_team.id == betting_match[0].away_team_id:
            betting_dict['away_match'] = betting_match[0]
        else:
            raise(Exception(
                'Betting data {} does not match any existing '.format(scraped_betting_odd) +
                'team/match combinations'
            ))

        return BettingOdds(**betting_dict)

    def __update_db_betting_odds(self, db_duplicates, scraped_duplicates):
        for idx, db_betting_odd in enumerate(db_duplicates):
            scraped_betting_odd = scraped_duplicates[idx]

            if (db_betting_odd.win_odds == scraped_betting_odd['win_odds'] and
               db_betting_odd.point_spread == scraped_betting_odd['point_spread']):
                continue

            db_betting_odd.win_odds = scraped_betting_odd['win_odds']
            db_betting_odd.point_spread = scraped_betting_odd['point_spread']
