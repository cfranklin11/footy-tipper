import sys
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, Team, BettingOdds


MIN_YEAR = 2000


def seed_betting_odds(session, teams, matches):
    betting_df = pd.read_csv(
        os.path.join(project_path, 'data/afl_betting.csv'),
        parse_dates=['full_date'],
        infer_datetime_format=True
    )
    betting_records = betting_df.to_dict('records')

    for betting_record in betting_records:
        betting_team = next(
            team for team in teams if team.name == betting_record['team']
        )
        betting_match = next(
            (match for match in matches
                if match.date == betting_record['full_date'].to_pydatetime() and
                match.venue == betting_record['venue'])
        )

        betting = {
            'win_odds': betting_record['win_odds'],
            'point_spread': betting_record['point_spread'],
            'team': betting_team
        }

        if betting_team.id == betting_match.home_team_id:
            betting['home_match'] = betting_match
        elif betting_team.id == betting_match.away_team_id:
            betting['away_match'] = betting_match
        else:
            raise('Betting data {} does not match any existing '.format(betting_record) +
                  'team/match combinations')

        session.add(BettingOdds(**betting))


def match_df():
    return pd.read_csv(
        os.path.join(project_path, 'data/ft_match_list.csv'),
        parse_dates=['full_date'],
        infer_datetime_format=True
    )


def seed_matches(session, teams):
    match_records = match_df().to_dict('records')

    for match_record in match_records:
        # Heroku has 10K limit on DB records for the free tier,
        #  so we have to limit how far back we go when saving match data.
        if match_record['full_date'] > datetime(MIN_YEAR, 1, 1):
            match = {
                'date': match_record['full_date'],
                'season_round': match_record['season_round'],
                'venue': match_record['venue'],
                'home_score': match_record['home_score'],
                'away_score': match_record['away_score'],
                'home_team': next(
                    team for team in teams if team.name == match_record['home_team']
                ),
                'away_team': next(
                    team for team in teams if team.name == match_record['away_team']
                )
            }
            session.add(Match(**match))


def seed_teams(session):
    df = match_df()
    team_names = df['home_team'].append(df['away_team']).drop_duplicates()
    teams = [Team(name=team_name) for team_name in team_names]
    session.add_all(teams)


def main():
    engine = create_engine(app.config['DATABASE_URL'])
    Session = sessionmaker(bind=engine)
    session = Session()

    seed_teams(session)
    saved_teams = session.query(Team).all()

    seed_matches(session, saved_teams)
    saved_matches = session.query(Match).all()

    seed_betting_odds(session, saved_teams, saved_matches)

    try:
        session.commit()
    except:
        print('Something went wrong, rolling back')
        session.rollback()
        raise
    finally:
        print('Closing session...')
        session.close()


if __name__ == '__main__':
    main()
